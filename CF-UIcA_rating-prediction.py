from __future__ import division

#==========================================================================

def run_experiment(args):
    import os
    # set environment variables for theano
    os.environ['THEANO_FLAGS'] = "lib.cnmem=" + str(args.mem) + ",device=gpu" + str(args.gpu)

    import inspect
    import shutil
    import time
    import logging
    import six
    import collections
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
    J_u, J_v = args.J_u, args.J_v
    lr = args.lr
    alpha = args.alpha
    weight_decay = args.weight_decay
    n_step = args.n_step
    lookahead = args.lookahead
    max_epoch = args.max_epoch
    batch_size_u, batch_size_v = args.batch_size_u, args.batch_size_v
    share_params = not args.no_share_params
    nonlin_enc = layers.get_nonlin(args.nonlin_enc)
    nonlin_dec = layers.get_nonlin(args.nonlin_dec)

#----------------------------------------------------------------
# Dataset
    dataset = cfdataset.CFdata(name=args.dataset, split=args.split)

    N_stars = dataset.N_stars

    N_u, N_v = dataset.N_users, dataset.N_items
    R_train = dataset.R_train                   # int (3 * N_train_rating)
    R_test = dataset.R_test                     # int (3 * N_test_rating)

    n_valid_split = np.int(dataset.N_train_rating / 20)
    train_valid_perm = np.random.permutation(dataset.N_train_rating)
    R_valid = R_train[:,train_valid_perm[:n_valid_split]]
    R_train = R_train[:,train_valid_perm[n_valid_split:]]

    R_matrix = dict()
    R_matrix['train'] = scipy.sparse.coo_matrix((R_train[2], (R_train[0], R_train[1])), shape=(N_u, N_v)).toarray().astype('int32')
    R_matrix['valid'] = scipy.sparse.coo_matrix((R_valid[2], (R_valid[0], R_valid[1])), shape=(N_u, N_v)).toarray().astype('int32')
    R_matrix['test'] = scipy.sparse.coo_matrix((R_test[2], (R_test[0], R_test[1])), shape=(N_u, N_v)).toarray().astype('int32')
    N_rating = dict()
    N_rating['train'] = dataset.N_train_rating - n_valid_split
    N_rating['valid'] = n_valid_split
    N_rating['test'] = dataset.N_test_rating

    logger.info("%d users, %d items" % (N_u, N_v))
    logger.info("%d training ratings, %d validation ratings, %d test ratings" % (N_rating['train'], N_rating['valid'], N_rating['test']))
    logger.info("%d-star scale" % N_stars)

#----------------------------------------------------------------
# numpy variables
    # encoded vectors
    np_enc_u_h = np.zeros((N_u, D_u), dtype=floatX)
    np_enc_v_h = np.zeros((N_v, D_v), dtype=floatX)

#----------------------------------------------------------------
# Symbolic variables
    sym_lr = T.fscalar('lr')
    sym_Ru = T.imatrix('Ru')
    sym_Rv = T.imatrix('Rv')
    sym_dr_Ru = T.fscalar('dr_Ru')
    sym_dr_Rv = T.fscalar('dr_Rv')
    sym_uid_origin = T.ivector('uid_origin')
    sym_uid_minibatch = T.ivector('uid_minibatch')
    sym_vid_origin = T.ivector('vid_origin')
    sym_vid_minibatch = T.ivector('vid_minibatch')
    sym_R_minibatch = T.ivector('R_minibatch')

#----------------------------------------------------------------
# Model setup (training model)
    logger.info("Setting up model ...")

    # Input layers
    l_in_Ru = ll.InputLayer((None, N_v), input_var=sym_Ru, name='l_in_Ru')
    l_in_Rv = ll.InputLayer((None, N_u), input_var=sym_Rv, name='l_in_Rv')
    l_in_uid_origin = ll.InputLayer((None,), input_var=sym_uid_origin, name='l_in_uid_origin')
    l_in_vid_origin = ll.InputLayer((None,), input_var=sym_vid_origin, name='l_in_vid_origin')
    l_in_uid_minibatch = ll.InputLayer((None,), input_var=sym_uid_minibatch, name='l_in_uid_minibatch')
    l_in_vid_minibatch = ll.InputLayer((None,), input_var=sym_vid_minibatch, name='l_in_vid_minibatch')

    # Dropout layers
    l_in_Ru = ll.DropoutLayer(l_in_Ru, p=sym_dr_Ru, rescale=False, name='Dropout-l_in_Ru')
    l_in_Rv = ll.DropoutLayer(l_in_Rv, p=sym_dr_Rv, rescale=False, name='Dropout-l_in_Rv')

    # User encoder model h(Ru)
    l_enc_u_h = layers.OneHotEncodeLayer(l_in_Ru, num_units=D_u, rank=J_u, num_hots=N_stars, share_params=share_params, nonlinearity=None, name='Dense-l_enc_u_h')
    l_enc_u_h = ll.NonlinearityLayer(l_enc_u_h, nonlinearity=nonlin_enc, name='Nonlin-l_enc_u_h')

    # Item encoder model h(Rv)
    l_enc_v_h = layers.OneHotEncodeLayer(l_in_Rv, num_units=D_v, rank=J_v, num_hots=N_stars, share_params=share_params, nonlinearity=None, name='Dense-l_enc_v_h')
    l_enc_v_h = ll.NonlinearityLayer(l_enc_v_h, nonlinearity=nonlin_enc, name='Nonlin-l_enc_v_h')

    # User decoder model s(h(Ru))
    l_dec_u_s = layers.OneHotDecodeLayer([l_enc_u_h, l_in_vid_origin, l_in_uid_minibatch], num_units=N_v, rank=J_u, num_hots=N_stars, share_params=share_params, nonlinearity=None, name='Dense-l_dec_u_s')

    # Item decoder model s(h(Rv))
    l_dec_v_s = layers.OneHotDecodeLayer([l_enc_v_h, l_in_uid_origin, l_in_vid_minibatch], num_units=N_u, rank=J_v, num_hots=N_stars, share_params=share_params, nonlinearity=None, name='Dense-l_dec_v_s')

    # Likelihood model p(R)
    l_uv_s = ll.ElemwiseSumLayer([l_dec_u_s, l_dec_v_s], name='l_uv_s')
    l_r = ll.NonlinearityLayer(l_uv_s, nonlinearity=ln.softmax, name='l_r')
    l_r_ordinal = ll.NonlinearityLayer(l_uv_s, nonlinearity=layers.log_ordinal_softmax, name='l_r_ordinal')

#----------------------------------------------------------------
# Likelihood and RMSE
    # training
    p_r_train, log_p_r_ordinal_train = ll.get_output([l_r, l_r_ordinal], deterministic=False)

    log_p_r = T.mean(parmesan.distributions.log_multinomial(sym_R_minibatch-1, p_r_train))
    R_minibatch_one_hot = lasagne.utils.one_hot(sym_R_minibatch, m=N_stars+1)[:,1:]
    log_p_r_ordinal = T.mean(T.sum(log_p_r_ordinal_train * R_minibatch_one_hot, axis=1))
    regularization = lasagne.regularization.regularize_network_params([l_r], lasagne.regularization.l2)
    cost_function = - (1.0 - alpha) * log_p_r - alpha * log_p_r_ordinal + weight_decay * regularization

    predicts_train = T.sum(p_r_train * T.shape_padleft(T.arange(1, 1+N_stars)), axis=1)
    SE_train = T.sum(T.sqr(T.cast(sym_R_minibatch, floatX) - predicts_train))

    # test
    sym_enc_u_h = T.fmatrix('enc_u_h')
    sym_enc_v_h = T.fmatrix('enc_v_h')
    enc_u_h_out, enc_v_h_out = ll.get_output([l_enc_u_h, l_enc_v_h], deterministic=True)
    p_r_test, = ll.get_output([l_r], inputs={l_enc_u_h:sym_enc_u_h, l_enc_v_h:sym_enc_v_h}, deterministic=True)

    predicts_test = T.sum(p_r_test * T.shape_padleft(T.arange(1, 1+N_stars)), axis=1)
    SE_test = T.sum(T.sqr(T.cast(sym_R_minibatch, floatX) - predicts_test))

#----------------------------------------------------------------
# Gradients
    clip_grad = 1
    max_norm = 5

    params = ll.get_all_params([l_r,], trainable=True)
    for p in params:
        logger.debug("%s: %s" % (p, p.get_value().shape))

    grads = T.grad(cost_function, params)
    mgrads = lasagne.updates.total_norm_constraint(grads, max_norm=max_norm)
    cgrads = [T.clip(g, -clip_grad, clip_grad) for g in mgrads]

    updates = lasagne.updates.adam(cgrads, params, beta1=0.9, beta2=0.999, epsilon=1e-4, learning_rate=sym_lr)

#----------------------------------------------------------------
# Compile
    # training function
    logger.info("Compiling train_model ...")
    train_model = theano.function(
            inputs=[sym_lr, sym_uid_origin, sym_uid_minibatch, sym_vid_origin, sym_vid_minibatch, sym_R_minibatch, sym_Ru, sym_Rv, sym_dr_Ru, sym_dr_Rv],
            outputs=[log_p_r, SE_train],
            updates=updates,
            )

    # encoders
    logger.info("Compiling encode_model ...")
    u_encode_model = theano.function(inputs=[sym_Ru], outputs=enc_u_h_out)
    v_encode_model = theano.function(inputs=[sym_Rv], outputs=enc_v_h_out)

    # test function
    logger.info("Compiling test_model ...")
    test_model = theano.function(
            inputs=[sym_uid_origin, sym_uid_minibatch, sym_vid_origin, sym_vid_minibatch, sym_R_minibatch, sym_enc_u_h, sym_enc_v_h],
            outputs=[SE_test],
            )

#----------------------------------------------------------------
# Predict function
    def predict(which_set='test'):
        assert which_set in ['valid', 'test']
        if which_set == 'valid':
            R_matrix_cond = R_matrix['train']
        else:
            R_matrix_cond = R_matrix['train'] + R_matrix['valid']

        # test statistics
        SE_epoch = 0
        n_pred_epoch = 0

        # preconpute hidden representation
        u_end = 0
        while u_end < N_u:
            u_start, u_end = u_end, min(u_end + batch_size_u, N_u)
            # create user mini-batch
            u_batch_ids = np.arange(u_start, u_end).astype('int32')
            # create conditionals
            Ru_minibatch = R_matrix_cond[u_batch_ids,:]
            # encode
            np_enc_u_h[u_batch_ids] = u_encode_model(Ru_minibatch)

        v_end = 0
        while v_end < N_v:
            v_start, v_end = v_end, min(v_end + batch_size_v, N_v)
            # create item mini-batch
            v_batch_ids = np.arange(v_start, v_end).astype('int32')
            # create conditionals
            Rv_minibatch = R_matrix_cond[:,v_batch_ids].T
            # encode
            np_enc_v_h[v_batch_ids] = v_encode_model(Rv_minibatch)

        # loop mini-batches
        u_end = 0
        while u_end < N_u:
            u_start, u_end = u_end, min(u_end + batch_size_u, N_u)
            v_end = 0
            while v_end < N_v:
                v_start, v_end = v_end, min(v_end + batch_size_v, N_v)
                # create user mini-batch and item mini-batch
                u_batch_ids = np.arange(u_start, u_end).astype('int32')
                v_batch_ids = np.arange(v_start, v_end).astype('int32')

                # get encoded vectors
                Ru_encoded = np_enc_u_h[u_batch_ids,:]
                Rv_encoded = np_enc_v_h[v_batch_ids,:]

                # create test samples mini-batch
                R_matrix_minibatch = R_matrix[which_set][np.ix_(u_batch_ids, v_batch_ids)]
                R_matrix_minibatch_sparse = scipy.sparse.coo_matrix(R_matrix_minibatch)

                # prepare user and item IDs needed
                uid_minibatch = R_matrix_minibatch_sparse.row
                vid_minibatch = R_matrix_minibatch_sparse.col
                R_minibatch = R_matrix_minibatch_sparse.data

                n_pred_step = R_minibatch.shape[0]
                if n_pred_step == 0:
                    continue

                uid_origin = u_batch_ids[uid_minibatch]
                vid_origin = v_batch_ids[vid_minibatch]

                SE_step, = test_model(uid_origin, uid_minibatch, vid_origin, vid_minibatch, R_minibatch, Ru_encoded, Rv_encoded)

                SE_epoch += SE_step
                n_pred_epoch += n_pred_step

        # print info after test finished
        assert n_pred_epoch == N_rating[which_set]
        RMSE_epoch = np.sqrt(SE_epoch/n_pred_epoch) / (N_stars / 5.0)
        logger.critical("Estimated  %s  RMSE = %f (%d %s ratings)" % (which_set, RMSE_epoch, n_pred_epoch, which_set))
        return RMSE_epoch

#----------------------------------------------------------------
# Training
    best_valid_result = np.inf
    best_model = None
    n_epocs_without_improvement = 0

    logger.warning("Training started.")
    # loop epoch
    for epoch in range(1, 1+max_epoch):
        epoch_start_time = time.time()

        # training statistics
        LL_epoch_train, SE_epoch_train= 0, 0
        n_pred_epoch_train = 0

        # loop mini-batches
        for step in range(n_step):
            # sample i and j
            #i = np.random.randint(N_u)
            #j = np.random.randint(N_v)
            threshold_u = np.int(0.2 * N_u)
            threshold_v = np.int(0.2 * N_v)
            i = np.random.randint(low=threshold_u, high=N_u-min(threshold_u, batch_size_u))
            j = np.random.randint(low=threshold_v, high=N_v-min(threshold_v, batch_size_v))

            # calculate mini-batch size
            Bi = min(batch_size_u, N_u - i)
            Bj = min(batch_size_v, N_v - j)

            # sample user mini-batch and item mini-batch
            u_batch_ids_train = np.random.choice(N_u, Bi, replace=False).astype('int32')
            v_batch_ids_train = np.random.choice(N_v, Bj, replace=False).astype('int32')

            # create conditionals
            Ru_minibatch_train = R_matrix['train'][u_batch_ids_train,:]
            Rv_minibatch_train = R_matrix['train'][:,v_batch_ids_train].T
            Ru_minibatch_train[:,v_batch_ids_train] = 0
            Rv_minibatch_train[:,u_batch_ids_train] = 0

            # calculate dropout rate
            dr_Ru = 1.0 - 1.0 * j / (N_v - Bj)
            dr_Rv = 1.0 - 1.0 * i / (N_u - Bi)

            # create training samples mini-batch
            R_matrix_minibatch_train = R_matrix['train'][np.ix_(u_batch_ids_train, v_batch_ids_train)]
            R_matrix_minibatch_sparse_train = scipy.sparse.coo_matrix(R_matrix_minibatch_train)

            # prepare user and item IDs needed
            uid_minibatch_train = R_matrix_minibatch_sparse_train.row
            vid_minibatch_train = R_matrix_minibatch_sparse_train.col
            R_minibatch_train = R_matrix_minibatch_sparse_train.data

            n_pred_step_train = R_minibatch_train.shape[0]
            if n_pred_step_train == 0:
                logger.warning('no training samples in current mini-batch.(i=%d, j=%d)' % (i, j))
                continue

            uid_origin_train = u_batch_ids_train[uid_minibatch_train]
            vid_origin_train = v_batch_ids_train[vid_minibatch_train]

            # update parameters and calculate likelihood and RMSE
            LL_step_train, SE_step_train = train_model(lr, uid_origin_train, uid_minibatch_train, vid_origin_train, vid_minibatch_train, R_minibatch_train, Ru_minibatch_train, Rv_minibatch_train, dr_Ru, dr_Rv)
            LL_epoch_train += LL_step_train * n_pred_step_train
            SE_epoch_train += SE_step_train
            n_pred_epoch_train += n_pred_step_train

        # print info after epoch finished
        LL_epoch_train /= n_pred_epoch_train
        RMSE_epoch_train = np.sqrt(SE_epoch_train/n_pred_epoch_train) / (N_stars / 5.0)

        epoch_end_time = time.time()
        logger.info("Epoch %d, Estimated training RMSE = %f, LL = %f (%d training ratings). Elapsed time %fs." % (epoch, RMSE_epoch_train, LL_epoch_train, n_pred_epoch_train, epoch_end_time-epoch_start_time))

        # validation
        RMSE_valid = predict('valid')

        # termination
        if RMSE_valid < best_valid_result:
            n_epocs_without_improvement = 0
            best_valid_result = RMSE_valid
            best_model = ll.get_all_param_values([l_r,], trainable=True)
            logger.debug("New best model found!")
        else:
            n_epocs_without_improvement += 1
            if n_epocs_without_improvement >= lookahead:
                ll.set_all_param_values([l_r,], best_model, trainable=True)
                if lr > 1e-5:
                    n_epocs_without_improvement = 0
                    lr /= 4
                    logger.warning("Learning rate = %f now." % lr)
                else:
                    logger.warning("Training finished.")
                    break

#----------------------------------------------------------------
# Test
    RMSE_test = predict('test')

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

    parser.add_argument('-mem', type=float, default=0.45,
        help="Theano lib.cnmem")

    parser.add_argument('-message', type=str, default="rating-prediction",
        help="Messages about the experiment")

    parser.add_argument('-loglv', type=str, default="debug",
        help="Log level")

    parser.add_argument('-dataset', type=str, default="ml-1m",
        help="Dataset name")

    parser.add_argument('-split', type=int, default=1,
        help="Which train-test split")

    parser.add_argument('-seed', type=int, default=12345,
        help="numpy random seed for reproducibility")

#----------------------------------------------------------------

    parser.add_argument('-lr', type=float, default=0.001,
        help="Learning rate")

    parser.add_argument('-alpha', type=float, default=1.,
        help="Weight of ordinal cost")

    parser.add_argument('-weight_decay', type=float, default=1e-4,
        help="Weight decay")

    parser.add_argument('-n_step', type=int, default=12,
        help="# steps in each \"epoch\"")

    parser.add_argument('-lookahead', type=int, default=50,
        help="# epochs lookahead")

    parser.add_argument('-max_epoch', type=int, default=99999,
        help="Max # epochs")

    parser.add_argument('-batch_size_u', type=int, default=1000,
        help="Mini-batch size of u (users)")

    parser.add_argument('-batch_size_v', type=int, default=1000,
        help="Mini-batch size of v (items)")

    parser.add_argument('-nonlin_dec', type=str, default="tanh",
        help="decoder nonlinearity")

    parser.add_argument('-nonlin_enc', type=str, default="tanh",
        help="encoder nonlinearity")

    parser.add_argument('-no_share_params', action='store_true', default=False,
        help="Remove share parameters in one hot weights")

#----------------------------------------------------------------

    parser.add_argument('-D_u', type=int, default=500,
        help="Dim of u")

    parser.add_argument('-J_u', type=int, default=0,
        help="Rank of hidden vectors of u. 0 for full rank (D_u)")

    parser.add_argument('-D_v', type=int, default=500,
        help="Dim of v")

    parser.add_argument('-J_v', type=int, default=0,
        help="Rank of hidden vectors of v. 0 for full rank (D_v)")

#----------------------------------------------------------------
    args = parser.parse_args()
    run_experiment(args)
