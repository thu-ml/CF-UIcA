from __future__ import division

#==========================================================================

def run_experiment(args):
    import os
    # set environment variables for theano
    os.environ['THEANO_FLAGS'] = "lib.cnmem=" + str(args.mem) + ",device=gpu" + str(args.gpu)

    import threading
    import Queue
    import inspect
    import shutil
    import time
    import logging
    import six
    import collections
    import itertools
    import random
    import numpy as np
    import scipy
    import theano
    import theano.tensor as T
    import lasagne
    import lasagne.layers as ll
    import lasagne.nonlinearities as ln
    import parmesan

    import layers
    import utils
    import cfdataset

#----------------------------------------------------------------
# Arguments and Settings
    floatX = theano.config.floatX
    logger = logging.getLogger()
    np.random.seed(args.seed)

    # copy file for reproducibility
    dirname = utils.setup_logging(args.message, args.loglv)
    script_src = os.path.abspath(inspect.getfile(inspect.currentframe()))
    script_dst = os.path.join(dirname, os.path.split(script_src)[1])
    shutil.copyfile(script_src, script_dst)

    # print arguments
    args_dict = collections.OrderedDict(sorted(vars(args).items()))
    for k, v in six.iteritems(args_dict):
        logger.info("  %20s: %s" % (k, v))

    # get arguments
    D_u, D_v = args.D_u, args.D_v
    lr = args.lr
    weight_decay = args.weight_decay
    lookahead = args.lookahead
    max_epoch = args.max_epoch
    batch_size_u, batch_size_v = args.batch_size_u, args.batch_size_v
    nonlin_enc = layers.get_nonlin(args.nonlin_enc)
    nonlin_dec = layers.get_nonlin(args.nonlin_dec)
    negative_ratio = args.negative_ratio

#----------------------------------------------------------------
# Dataset
    dataset = cfdataset.CF_implicit_data(name=args.dataset)

    N_u, N_v = dataset.N_users, dataset.N_items
    T_matrix = dataset.T_matrix.astype(floatX)
    R_matrix = dataset.R_matrix.astype(floatX)
    R_negative_matrix = 1 - R_matrix
    assert np.all(R_matrix == (T_matrix > 0.5))
    assert np.all((R_negative_matrix == 1) == (T_matrix == 0))

    R_test = dataset.R_latest
    T_matrix[np.arange(N_u), R_test] = 0
    R_matrix[np.arange(N_u), R_test] = 0
    assert np.all(R_matrix == (T_matrix > 0.5))

    R_matrix_for_test = R_matrix.copy()

    R_valid = dataset.R_2nd_latest
    T_matrix[np.arange(N_u), R_valid] = 0
    R_matrix[np.arange(N_u), R_valid] = 0
    assert np.all(R_matrix == (T_matrix > 0.5))

    N_interaction = dataset.N_interaction - N_u * 2

    assert np.all(R_valid != R_test)
    assert np.all(R_matrix_for_test[np.arange(N_u), R_valid] == 1)
    assert np.all(R_matrix_for_test[np.arange(N_u), R_test] == 0)
    assert np.all(R_matrix[np.arange(N_u), R_valid] == 0)
    assert np.all(R_matrix[np.arange(N_u), R_test] == 0)
    assert np.all(T_matrix[np.arange(N_u), R_valid] == 0)
    assert np.all(T_matrix[np.arange(N_u), R_test] == 0)
    assert N_interaction == np.count_nonzero(R_matrix)
    assert N_interaction + N_u == np.count_nonzero(R_matrix_for_test)

    logger.info("%d users, %d items, %d training interactions (%d total, 2 * %d held out for validation and test)." % (N_u, N_v, N_interaction, dataset.N_interaction, N_u))

#----------------------------------------------------------------
# numpy variables
    # encoded vectors
    np_enc_u_h = np.zeros((N_u, D_u), dtype=floatX)
    np_enc_v_h = np.zeros((N_v, D_v), dtype=floatX)

#----------------------------------------------------------------
# Symbolic variables
    sym_lr = T.fscalar('lr')

    sym_Ru_pos = T.fmatrix('Ru_pos')
    sym_dr_Ru_pos = T.fscalar('dr_Ru_pos')
    sym_uid_origin_pos = T.ivector('uid_origin_pos')
    sym_uid_minibatch_pos = T.ivector('uid_minibatch_pos')

    sym_Ru_neg = T.fmatrix('Ru_neg')
    sym_dr_Ru_neg = T.fscalar('dr_Ru_neg')
    sym_uid_origin_neg = T.ivector('uid_origin_neg')
    sym_uid_minibatch_neg = T.ivector('uid_minibatch_neg')

    sym_Rv = T.fmatrix('Rv')
    sym_dr_Rv = T.fscalar('dr_Rv')
    sym_vid_origin_pos = T.ivector('vid_origin_pos')
    sym_vid_minibatch_pos = T.ivector('vid_minibatch_pos')
    sym_vid_origin_neg = T.ivector('vid_origin_neg')
    sym_vid_minibatch_neg = T.ivector('vid_minibatch_neg')

    sym_R_minibatch = T.fvector('R_minibatch')

#----------------------------------------------------------------
# Model setup (training model)
    logger.info("Setting up model ...")

    # Input layers
    l_in_Ru_pos = ll.InputLayer((None, N_v), input_var=sym_Ru_pos, name='l_in_Ru_pos')
    l_in_uid_origin_pos = ll.InputLayer((None,), input_var=sym_uid_origin_pos, name='l_in_uid_origin_pos')
    l_in_uid_minibatch_pos = ll.InputLayer((None,), input_var=sym_uid_minibatch_pos, name='l_in_uid_minibatch_pos')

    l_in_Ru_neg = ll.InputLayer((None, N_v), input_var=sym_Ru_neg, name='l_in_Ru_neg')
    l_in_uid_origin_neg = ll.InputLayer((None,), input_var=sym_uid_origin_neg, name='l_in_uid_origin_neg')
    l_in_uid_minibatch_neg = ll.InputLayer((None,), input_var=sym_uid_minibatch_neg, name='l_in_uid_minibatch_neg')

    l_in_Rv = ll.InputLayer((None, N_u), input_var=sym_Rv, name='l_in_Rv')
    l_in_vid_origin_pos = ll.InputLayer((None,), input_var=sym_vid_origin_pos, name='l_in_vid_origin_pos')
    l_in_vid_minibatch_pos = ll.InputLayer((None,), input_var=sym_vid_minibatch_pos, name='l_in_vid_minibatch_pos')
    l_in_vid_origin_neg = ll.InputLayer((None,), input_var=sym_vid_origin_neg, name='l_in_vid_origin_neg')
    l_in_vid_minibatch_neg = ll.InputLayer((None,), input_var=sym_vid_minibatch_neg, name='l_in_vid_minibatch_neg')

    # Dropout layers
    l_in_Ru_pos = ll.DropoutLayer(l_in_Ru_pos, p=sym_dr_Ru_pos, rescale=False, name='Dropout-l_in_Ru_pos')
    l_in_Ru_neg = ll.DropoutLayer(l_in_Ru_neg, p=sym_dr_Ru_neg, rescale=False, name='Dropout-l_in_Ru_neg')
    l_in_Rv = ll.DropoutLayer(l_in_Rv, p=sym_dr_Rv, rescale=False, name='Dropout-l_in_Rv')

    # User encoder model h(Ru)
    l_enc_u_h_pos = ll.DenseLayer(l_in_Ru_pos, num_units=D_u, nonlinearity=nonlin_enc, name='l_enc_u_h_pos')
    l_enc_u_h_neg = ll.DenseLayer(l_in_Ru_neg, num_units=D_u, nonlinearity=nonlin_enc, W=l_enc_u_h_pos.W, b=l_enc_u_h_pos.b, name='l_enc_u_h_neg')

    # Item encoder model h(Rv)
    l_enc_v_h = ll.DenseLayer(l_in_Rv, num_units=D_v, nonlinearity=nonlin_enc, name='l_enc_v_h')

    # User decoder model s(h(Ru))
    l_dec_u_s_pos = layers.SimpleDecodeLayer([l_enc_u_h_pos, l_in_vid_origin_pos, l_in_uid_minibatch_pos], num_units=N_v, nonlinearity=None, name='l_dec_u_s_pos')
    l_dec_u_s_neg = layers.SimpleDecodeLayer([l_enc_u_h_neg, l_in_vid_origin_neg, l_in_uid_minibatch_neg], num_units=N_v, V=l_dec_u_s_pos.V, Q=l_dec_u_s_pos.Q, b=l_dec_u_s_pos.b, nonlinearity=None, name='l_dec_u_s_neg')
    l_dec_u_s_all = ll.ConcatLayer([l_dec_u_s_pos ,l_dec_u_s_neg], axis=0)

    # Item decoder model s(h(Rv))
    l_dec_v_s_pos = layers.SimpleDecodeLayer([l_enc_v_h, l_in_uid_origin_pos, l_in_vid_minibatch_pos], num_units=N_u, nonlinearity=None, name='l_dec_v_s_pos')
    l_dec_v_s_neg = layers.SimpleDecodeLayer([l_enc_v_h, l_in_uid_origin_neg, l_in_vid_minibatch_neg], num_units=N_u, V=l_dec_v_s_pos.V, Q=l_dec_v_s_pos.Q, b=l_dec_v_s_pos.b, nonlinearity=None, name='l_dec_v_s_neg')
    l_dec_v_s_all = ll.ConcatLayer([l_dec_v_s_pos ,l_dec_v_s_neg], axis=0)

    # Likelihood model p(R)
    l_uv_s_train = ll.ElemwiseSumLayer([l_dec_u_s_all, l_dec_v_s_all], name='l_uv_s_train')
    l_r_train = ll.NonlinearityLayer(l_uv_s_train, nonlinearity=ln.sigmoid, name='l_r_train')
    l_uv_s_test = ll.ElemwiseSumLayer([l_dec_u_s_pos, l_dec_v_s_pos], name='l_uv_s_test')
    l_r_test = ll.NonlinearityLayer(l_uv_s_test, nonlinearity=ln.sigmoid, name='l_r_test')

#----------------------------------------------------------------
# Likelihood and RMSE
    # training
    p_r_train, = ll.get_output([l_r_train], deterministic=False)

    log_p_r = T.mean(parmesan.distributions.log_bernoulli(sym_R_minibatch, p_r_train, eps=1e-6))
    regularization = lasagne.regularization.regularize_network_params([l_r_train], lasagne.regularization.l2)
    cost_function = - log_p_r + weight_decay * regularization

    SE_train = T.sum(T.sqr(sym_R_minibatch - p_r_train))

    # test
    sym_enc_u_h = T.fmatrix('enc_u_h')
    sym_enc_v_h = T.fmatrix('enc_v_h')
    enc_u_h_out, enc_v_h_out = ll.get_output([l_enc_u_h_pos, l_enc_v_h], deterministic=True)
    p_r_test, = ll.get_output([l_r_test], inputs={l_enc_u_h_pos:sym_enc_u_h, l_enc_v_h:sym_enc_v_h}, deterministic=True)
    test_scores = p_r_test.reshape((-1, 101))
    ranking = test_scores.argsort()[:,::-1].argmin(axis=1)

#----------------------------------------------------------------
# Gradients
    clip_grad = 1
    max_norm = 5

    params = ll.get_all_params([l_r_train,], trainable=True)
    for p in params:
        logger.debug("%s: %s" % (p, p.get_value().shape))

    grads = T.grad(cost_function, params)
    mgrads = lasagne.updates.total_norm_constraint(grads, max_norm=max_norm)
    cgrads = [T.clip(g, -clip_grad, clip_grad) for g in mgrads]

    #updates = lasagne.updates.adam(cgrads, params, beta1=0.9, beta2=0.999, epsilon=1e-4, learning_rate=sym_lr)
    updates, sym_vars_list = utils.adam(cgrads, params, beta1=0.9, beta2=0.999, epsilon=1e-4, learning_rate=sym_lr)

    # moving average
    params_avg=[]
    for param in params:
        value = param.get_value(borrow=True)
        params_avg.append(theano.shared(np.zeros(value.shape, dtype=value.dtype),
                              broadcastable=param.broadcastable,
                              name=param.name + '_avg'))
    avg_updates = [(a, a + 0.01 * (p - a)) for p, a in zip(params, params_avg)]
    avg_givens = [(p, a) for p, a in zip(params, params_avg)]
    all_updates = updates.items() + avg_updates

#----------------------------------------------------------------
# Compile
    # training function
    logger.info("Compiling train_model ...")
    train_model = theano.function(
            inputs=[sym_lr,
                sym_uid_origin_pos, sym_uid_minibatch_pos, sym_vid_origin_pos, sym_vid_minibatch_pos,
                sym_uid_origin_neg, sym_uid_minibatch_neg, sym_vid_origin_neg, sym_vid_minibatch_neg,
                sym_Ru_pos, sym_Ru_neg, sym_Rv,
                sym_R_minibatch, sym_dr_Ru_pos, sym_dr_Ru_neg, sym_dr_Rv],
            outputs=[log_p_r, SE_train],
            updates=all_updates,
            )

    # encoders
    logger.info("Compiling encode_model ...")
    u_encode_model = theano.function(inputs=[sym_Ru_pos], outputs=enc_u_h_out)
    v_encode_model = theano.function(inputs=[sym_Rv], outputs=enc_v_h_out)

    u_encode_avg_model = theano.function(inputs=[sym_Ru_pos], outputs=enc_u_h_out, givens=avg_givens, on_unused_input='ignore')
    v_encode_avg_model = theano.function(inputs=[sym_Rv], outputs=enc_v_h_out, givens=avg_givens, on_unused_input='ignore')

    # test function
    logger.info("Compiling test_model ...")
    test_model = theano.function(
            inputs=[sym_uid_origin_pos, sym_uid_minibatch_pos, sym_vid_origin_pos, sym_vid_minibatch_pos, sym_enc_u_h, sym_enc_v_h],
            outputs=[ranking],
            )

    test_avg_model = theano.function(
            inputs=[sym_uid_origin_pos, sym_uid_minibatch_pos, sym_vid_origin_pos, sym_vid_minibatch_pos, sym_enc_u_h, sym_enc_v_h],
            outputs=[ranking],
            givens=avg_givens, on_unused_input='ignore',
            )

#----------------------------------------------------------------
# Predict function
    def compute_hidden_for(for_which_set='test', avg_model=False):
        assert for_which_set in ['valid', 'test']
        if for_which_set == 'valid':
            R_matrix_cond = R_matrix
        else:
            R_matrix_cond = R_matrix_for_test

        # preconpute hidden representation
        u_end = 0
        while u_end < N_u:
            u_start, u_end = u_end, min(u_end + batch_size_u, N_u)
            # create user mini-batch
            u_batch_ids = np.arange(u_start, u_end).astype('int32')
            # create conditionals
            Ru_minibatch = R_matrix_cond[u_batch_ids,:]
            # encode
            if avg_model:
                np_enc_u_h[u_batch_ids] = u_encode_avg_model(Ru_minibatch)
            else:
                np_enc_u_h[u_batch_ids] = u_encode_model(Ru_minibatch)

        v_end = 0
        while v_end < N_v:
            v_start, v_end = v_end, min(v_end + batch_size_v, N_v)
            # create item mini-batch
            v_batch_ids = np.arange(v_start, v_end).astype('int32')
            # create conditionals
            Rv_minibatch = R_matrix_cond[:,v_batch_ids].T
            # encode
            if avg_model:
                np_enc_v_h[v_batch_ids] = v_encode_avg_model(Rv_minibatch)
            else:
                np_enc_v_h[v_batch_ids] = v_encode_model(Rv_minibatch)

    def predict_once(which_set='test', avg_model=False):
        assert which_set in ['valid', 'test']
        if which_set == 'valid':
            R_predict = R_valid
        else:
            R_predict = R_test

        # test statistics
        rankings = []

        # loop users
        u_end = 0
        while u_end < N_u:
            u_start, u_end = u_end, min(u_end + batch_size_u, N_u)

            # create user mini-batch and item mini-batch
            u_batch_ids = np.arange(u_start, u_end).astype('int32')

            vid_negative = np.asarray([np.random.choice(np.where(row)[0], 100, replace=False) for row in R_negative_matrix[u_batch_ids]], dtype='int32')
            vid = np.concatenate([R_predict[u_batch_ids].reshape(-1,1), vid_negative], axis=1).flatten()
            uid_origin = np.repeat(u_batch_ids, 101)
            uid_minibatch = uid_origin - u_start

            # get encoded vectors
            Ru_encoded = np_enc_u_h[u_batch_ids]

            if avg_model:
                rankings_minibatch, = test_avg_model(uid_origin, uid_minibatch, vid, vid, Ru_encoded, np_enc_v_h)
            else:
                rankings_minibatch, = test_model(uid_origin, uid_minibatch, vid, vid, Ru_encoded, np_enc_v_h)
            rankings.append(rankings_minibatch)

        rankings = np.concatenate(rankings)
        HR = np.mean(rankings < 10)
        NDCG = np.mean((rankings < 10) / np.log2(rankings + 2))

        return HR, NDCG

    def predict(which_set='test', avg=10, avg_model=False):
        compute_hidden_for(for_which_set=which_set, avg_model=avg_model)
        HR_list = []
        NDCG_list = []
        for i in range(avg):
            hr, ndcg = predict_once(which_set=which_set, avg_model=avg_model)
            HR_list.append(hr)
            NDCG_list.append(ndcg)
        HR_mean = np.mean(HR_list)
        NDCG_mean = np.mean(NDCG_list)
        HR_std = np.std(HR_list)
        NDCG_std = np.std(NDCG_list)
        # print info after test finished
        eval_msg = which_set if not avg_model else which_set + ' (avg model)'
        logger.critical("%-20s HR = %.3f +- %.3f, NDCG = %.3f +- %.3f." % (eval_msg, HR_mean, HR_std, NDCG_mean, NDCG_std))
        return HR_mean, NDCG_mean

#----------------------------------------------------------------
# Training
    best_valid_result = - np.inf
    best_model = None
    best_auxiliary = None
    n_epocs_without_improvement = 0

    minibatch_queue = Queue.Queue(maxsize=10)

    # function for preparing minibatches
    def prepare_minibatch(minibatch_list):
        # loop mini-batches
        for u_batch_ids, v_batch_ids in minibatch_list:
            Rv_minibatch = R_matrix[:,v_batch_ids].T
            Rv_minibatch[:,u_batch_ids] = 0
            Ru_minibatch_neg = R_matrix[u_batch_ids,:]
            #Ru_minibatch_neg[:,v_batch_ids] = 0

            # create training samples mini-batch
            T_matrix_minibatch = T_matrix[np.ix_(u_batch_ids, v_batch_ids)]
            T_matrix_minibatch_sparse = scipy.sparse.coo_matrix(T_matrix_minibatch)
            n_interactions_minibatch = T_matrix_minibatch_sparse.count_nonzero()
            Ru_minibatch_pos = ((T_matrix[u_batch_ids[T_matrix_minibatch_sparse.row]] < T_matrix_minibatch_sparse.data.reshape(n_interactions_minibatch, 1)) & (T_matrix[u_batch_ids[T_matrix_minibatch_sparse.row]] > 0)).astype(floatX)

            uid_minibatch_pos = np.arange(n_interactions_minibatch).astype('int32')
            uid_origin_pos = u_batch_ids[T_matrix_minibatch_sparse.row]
            vid_minibatch_pos = T_matrix_minibatch_sparse.col
            vid_origin_pos = v_batch_ids[vid_minibatch_pos]

            R_matrix_negative_minibatch = 1 - R_matrix[np.ix_(u_batch_ids, v_batch_ids)]
            R_matrix_negative_minibatch_sparse = scipy.sparse.coo_matrix(R_matrix_negative_minibatch)
            n_negative_total = R_matrix_negative_minibatch_sparse.count_nonzero()
            assert n_negative_total + n_interactions_minibatch == u_batch_ids.size * v_batch_ids.size
            choice_negative = np.random.choice(n_negative_total, min(n_negative_total, np.int(n_interactions_minibatch * negative_ratio)), replace=False)

            uid_minibatch_neg = R_matrix_negative_minibatch_sparse.row[choice_negative]
            uid_origin_neg = u_batch_ids[uid_minibatch_neg]
            vid_minibatch_neg = R_matrix_negative_minibatch_sparse.col[choice_negative]
            vid_origin_neg = v_batch_ids[vid_minibatch_neg]

            R_minibatch = np.concatenate([np.ones_like(T_matrix_minibatch_sparse.data), R_matrix_negative_minibatch_sparse.data[choice_negative] * 0])

            n_pred_step = R_minibatch.shape[0]
            if n_pred_step == 0:
                raise ValueError('No interactions in this minibatch.')

            dr_Ru_pos = min(max(1 - 2 * np.random.rand(), 0), 0.8)
            dr_Ru_neg = 0.2
            dr_Rv = min(max(1 - 2 * np.random.rand(), 0), 0.8)

            # package everything into a tuple
            data_minibatch_package = (
                    uid_origin_pos, uid_minibatch_pos, vid_origin_pos, vid_minibatch_pos,
                    uid_origin_neg, uid_minibatch_neg, vid_origin_neg, vid_minibatch_neg,
                    Ru_minibatch_pos, Ru_minibatch_neg, Rv_minibatch,
                    R_minibatch, dr_Ru_pos, dr_Ru_neg, dr_Rv)

            # enqueue
            minibatch_queue.put((n_pred_step, data_minibatch_package))

    logger.warning("Training started.")
    # loop epoch
    for epoch in range(1, 1+max_epoch):
        epoch_start_time = time.time()

        # training statistics
        LL_epoch, SE_epoch= 0, 0
        n_pred_epoch = 0

        u_order = np.array_split(np.random.permutation(N_u).astype('int32'), N_u // batch_size_u + 1)
        v_order = np.array_split(np.random.permutation(N_v).astype('int32'), N_v // batch_size_v + 1)
        minibatch_order = list(itertools.product(u_order, v_order))
        random.shuffle(minibatch_order)

        n_threads = 5
        n_minibatch_thread = len(minibatch_order) // n_threads + 1
        for t in range(n_threads):
            thr = threading.Thread(target=prepare_minibatch, args=(minibatch_order[t*n_minibatch_thread:(t+1)*n_minibatch_thread],))
            thr.setDaemon(True)
            thr.start()

        for step in range(len(minibatch_order)):
            n_pred_step, data_minibatch_package = minibatch_queue.get()
            # update parameters and calculate likelihood and RMSE
            LL_step, SE_step = train_model(lr, *data_minibatch_package)
            minibatch_queue.task_done()
            LL_epoch += LL_step * n_pred_step
            SE_epoch += SE_step
            n_pred_epoch += n_pred_step

        assert minibatch_queue.qsize() == 0

        # print info after epoch finished
        LL_epoch /= n_pred_epoch
        RMSE_epoch = np.sqrt(SE_epoch/n_pred_epoch)

        epoch_end_time = time.time()
        logger.info("Epoch %d, training RMSE = %f, LL = %f (%d training ratings). Elapsed time %.1fs." % (epoch, RMSE_epoch, LL_epoch, n_pred_epoch, epoch_end_time-epoch_start_time))

        # validation
        HR_valid, NDCG_valid = predict('valid')
        HR_test, NDCG_test = predict('test')
        HR_test, NDCG_test = predict('test', avg_model=True)

        # termination
        #if NDCG_valid > best_valid_result:
        if HR_valid > best_valid_result:
            n_epocs_without_improvement = 0
            #best_valid_result = NDCG_valid
            best_valid_result = HR_valid
            best_model = ll.get_all_param_values([l_r_train,], trainable=True)
            best_auxiliary = utils.get_all_shvar_values(sym_vars_list)
            logger.debug("New best model found!")
        else:
            n_epocs_without_improvement += 1
            if n_epocs_without_improvement >= lookahead:
                ll.set_all_param_values([l_r_train,], best_model, trainable=True)
                utils.set_all_shvar_values(sym_vars_list, best_auxiliary)
                if lr > 1e-5:
                    n_epocs_without_improvement = 0
                    lr /= 4
                    logger.error("Learning rate = %f now." % lr)
                else:
                    logger.error("Training finished.")
                    break

#----------------------------------------------------------------
# Test
    HR_test, NDCG_test = predict('test')
    HR_test, NDCG_test = predict('test', avg_model=True)

#----------------------------------------------------------------
# Summarization
    for k, v in six.iteritems(args_dict):
        logger.info("  %20s: %s" % (k, v))

#==========================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-gpu', type=int, default=0,
        help="ID of the gpu device to use")

    parser.add_argument('-mem', type=float, default=0.95,
        help="Theano lib.cnmem")

    parser.add_argument('-message', type=str, default="topN-recommendation",
        help="Messages about the experiment")

    parser.add_argument('-loglv', type=str, default="debug",
        help="Log level")

    parser.add_argument('-dataset', type=str, default="ml-1m",
        help="Dataset name")

    parser.add_argument('-seed', type=int, default=12345,
        help="numpy random seed for reproducibility")

#----------------------------------------------------------------

    parser.add_argument('-lr', type=float, default=0.001,
        help="Learning rate")

    parser.add_argument('-weight_decay', type=float, default=1e-5,
        help="Weight decay")

    parser.add_argument('-lookahead', type=int, default=30,
        help="# epochs lookahead")

    parser.add_argument('-max_epoch', type=int, default=99999,
        help="Max # epochs")

    parser.add_argument('-batch_size_u', type=int, default=200,
        help="Mini-batch size of u (users)")

    parser.add_argument('-batch_size_v', type=int, default=200,
        help="Mini-batch size of v (items)")

    parser.add_argument('-nonlin_dec', type=str, default="rectify",
        help="decoder nonlinearity")

    parser.add_argument('-nonlin_enc', type=str, default="rectify",
        help="encoder nonlinearity")

    parser.add_argument('-negative_ratio', type=float, default=4.0,
        help="Ratio of negative interactions")

#----------------------------------------------------------------

    parser.add_argument('-D_u', type=int, default=256,
        help="Dim of u")

    parser.add_argument('-D_v', type=int, default=256,
        help="Dim of v")

#----------------------------------------------------------------
    args = parser.parse_args()
    run_experiment(args)
