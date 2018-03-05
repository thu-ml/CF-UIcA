import numpy as np
import theano
import theano.tensor as T
import lasagne
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

floatX = theano.config.floatX

#----------------------------------------------------------------

def get_nonlin(nonlin):
    if nonlin == 'rectify':
        return lasagne.nonlinearities.rectify
    elif nonlin == 'very_leaky_rectify':
        return lasagne.nonlinearities.very_leaky_rectify
    elif nonlin == 'tanh':
        return lasagne.nonlinearities.tanh
    elif nonlin == 'sigmoid':
        return lasagne.nonlinearities.sigmoid
    elif nonlin == 'softmax':
        return lasagne.nonlinearities.softmax
    else:
        raise ValueError('invalid non-linearity \'' + nonlin + '\'')

#----------------------------------------------------------------

def log_ordinal_softmax(S):
    activation = T.nnet.softmax(S) + 1e-4
    left_denominator = theano.tensor.extra_ops.cumsum(activation, axis=1)
    right_denominator = theano.tensor.extra_ops.cumsum(activation[:,::-1], axis=1)[:,::-1]
    left_fraction = T.log(activation) - T.log(left_denominator)
    right_fraction = T.log(activation) - T.log(right_denominator)
    left = theano.tensor.extra_ops.cumsum(left_fraction, axis=1)
    right = theano.tensor.extra_ops.cumsum(right_fraction[:,::-1], axis=1)[:,::-1]
    return left + right

#----------------------------------------------------------------

class OneHotEncodeLayer(lasagne.layers.Layer):
    '''
    Accept a raw rating input, convert it to one-hot representation and output its activation.
    The one-hot convertion is handled inside the layer.
    '''
    def __init__(self, incoming,
            num_units,
            rank=0,
            num_hots=5,
            share_params=False,
            W=lasagne.init.GlorotUniform(),
            P=lasagne.init.GlorotUniform(),
            c=lasagne.init.Constant(0.),
            nonlinearity=lasagne.nonlinearities.rectify,
            **kwargs):
        super(OneHotEncodeLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (lasagne.nonlinearities.identity if nonlinearity is None else nonlinearity)
        self.num_units = num_units
        self.rank = num_units if rank <= 0 else min(rank, num_units)
        assert self.rank > 0
        self.num_hots = num_hots
        self.share_params = share_params
        self.W = self.add_param(W, (self.rank, self.input_shape[1], num_hots), name='W')
        if self.rank == num_units:
            self.P = None
        else:
            self.P = self.add_param(P, (self.rank, num_units), name='P')
        if c is None:
            self.c = None
        else:
            self.c = self.add_param(c, (num_units,), name='c', regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        base_matrix = T.eye(self.num_hots+1)
        if self.share_params:
            base_matrix = T.extra_ops.cumsum(base_matrix, axis=0)
        base_matrix = base_matrix[:,1:]
        input_one_hot = base_matrix[input]
        y = T.tensordot(input_one_hot, self.W, [[1,2],[1,2]])
        if self.P is not None:
            y = y.dot(self.P)
        y = y + self.c.dimshuffle('x',0)
        y = self.nonlinearity(y)
        return y

class OneHotDecodeLayer(lasagne.layers.MergeLayer):
    def __init__(self, incomings,
            num_units,
            rank=0,
            num_hots=5,
            share_params=False,
            V=lasagne.init.GlorotUniform(),
            Q=lasagne.init.GlorotUniform(),
            b=lasagne.init.Constant(0.),
            nonlinearity=lasagne.nonlinearities.rectify,
            **kwargs):
        super(OneHotDecodeLayer, self).__init__(incomings, **kwargs)
        self.nonlinearity = (lasagne.nonlinearities.identity if nonlinearity is None else nonlinearity)
        self.num_units = num_units
        self.hidden_dims = self.input_shapes[0][1]
        self.rank = self.hidden_dims if rank <= 0 else min(rank, self.hidden_dims)
        assert self.rank > 0
        self.num_hots = num_hots
        self.share_params = share_params
        self.V = self.add_param(V, (num_units, self.rank, num_hots), name='V')
        if self.rank == self.hidden_dims:
            self.Q = None
        else:
            self.Q = self.add_param(Q, (self.hidden_dims, self.rank), name='Q')
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units, num_hots), name='b', regularizable=False)

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[1][0], self.num_hots)

    def get_output_for(self, inputs, **kwargs):
        h, id_origin, id_minibatch = inputs
        if self.Q is not None:
            h = h.dot(self.Q)
        h_all = h[id_minibatch]
        V_all = self.V[id_origin]
        y = T.sum(V_all * T.shape_padright(h_all), axis=1)
        y = y + self.b[id_origin]
        if self.share_params:
            y = T.extra_ops.cumsum(y, axis=1)
        y = self.nonlinearity(y)
        return y

class SimpleDecodeLayer(lasagne.layers.MergeLayer):
    def __init__(self, incomings,
            num_units,
            rank=0,
            V=lasagne.init.GlorotUniform(),
            Q=lasagne.init.GlorotUniform(),
            b=lasagne.init.Constant(0.),
            nonlinearity=lasagne.nonlinearities.rectify,
            **kwargs):
        super(SimpleDecodeLayer, self).__init__(incomings, **kwargs)
        self.nonlinearity = (lasagne.nonlinearities.identity if nonlinearity is None else nonlinearity)
        self.num_units = num_units
        self.hidden_dims = self.input_shapes[0][1]
        self.rank = self.hidden_dims if rank <= 0 else min(rank, self.hidden_dims)
        assert self.rank > 0
        self.V = self.add_param(V, (num_units, self.rank), name='V')
        if self.rank == self.hidden_dims:
            self.Q = None
        else:
            self.Q = self.add_param(Q, (self.hidden_dims, self.rank), name='Q')
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units,), name='b', regularizable=False)

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[1][0],)

    def get_output_for(self, inputs, **kwargs):
        h, id_origin, id_minibatch = inputs
        if self.Q is not None:
            h = h.dot(self.Q)
        h_all = h[id_minibatch]
        V_all = self.V[id_origin]
        y = T.sum(V_all * h_all, axis=1)
        if self.b is not None:
            y = y + self.b[id_origin]
        y = self.nonlinearity(y)
        return y
