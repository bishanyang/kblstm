'''
Created on Jul 24, 2016

@author: bishan
'''
import numpy as np

from collections import OrderedDict

import theano
import theano.tensor.shared_randomstreams
import theano.tensor as T

theano.config.floatX = 'float32'
rng = np.random.RandomState(3435)

def shared(shape, name):
    """
    Create a shared object of a numpy array.
    """
    if len(shape) == 1:
        value = np.zeros(shape)  # bias are initialized with zeros
    else:
        drange = np.sqrt(6. / (np.sum(shape)))
        value = drange * rng.uniform(low=-1.0, high=1.0, size=shape)
           
    return theano.shared(value=value.astype(theano.config.floatX), name=name)

def orth_normal_initializer(factor=1.0, seed=None):
    ''' Reference: Exact solutions to the nonlinear dynamics of learning in
                   deep linear neural networks
          Saxe et al., 2014. https://arxiv.org/pdf/1312.6120.pdf
        Adapted from the original implementation by Mingxuan Wang.
    '''
    def _initializer(shape, dtype):
        assert len(shape) == 2
        rng = np.random.RandomState(seed)
        if shape[0] == shape[1]:
            M = rng.randn(*shape).astype(dtype)
            Q, R = np.linalg.qr(M)
            Q = Q * np.sign(np.diag(R))
            param = Q * factor
            return param
        else:
            M1 = rng.randn(shape[0], shape[0]).astype(dtype)
            M2 = rng.randn(shape[1], shape[1]).astype(dtype)
            Q1, R1 = np.linalg.qr(M1)
            Q2, R2 = np.linalg.qr(M2)
            Q1 = Q1 * np.sign(np.diag(R1))
            Q2 = Q2 * np.sign(np.diag(R2))
            n_min = min(shape[0], shape[1])
            param = np.dot(Q1[:, :n_min], Q2[:n_min, :]) * factor
            return param
    return _initializer

def block_orth_normal_initializer(shape, input_shapes, output_shapes, name):
    ''' Initialize a gigantic weight matrix where each block is a normal orthogonal matrix.
      Input:
        - input_shapes: the sizes of each block alone dimension 0.
        - output_shapes: the sizes of each block along dimension 1.
        for example input_shapes = [100, 128] output_shapes=[100,100,100,100]
          indicates eight blocks with shapes [100,100], [128,100], etc.
    '''
    assert len(shape) == 2
    initializer = orth_normal_initializer()
    params = np.concatenate([np.concatenate([initializer([dim_in, dim_out], dtype=theano.config.floatX)
             for dim_out in output_shapes], 1)
            for dim_in in input_shapes], 0)
    return theano.shared(value=params, name=name, borrow=True)

def init_param(rng, shape, is_empty=False):
    if is_empty:
        return np.zeros(shape, dtype=theano.config.floatX)
    else:
        return rng.normal(scale=0.1, size=shape).astype(theano.config.floatX)

def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)
    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)

def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9,word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param      
    return updates 

def sgd_updates_adagrad(params, hist_grads, cost, learning_rate):
    # adagrad
    eps = 1e-8 
    
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})

    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        up_exp_sg = exp_sg + T.sqr(gp)
        updates[exp_sg] = up_exp_sg

        stepped_param = param - learning_rate / (eps + T.sqrt(up_exp_sg)) * gp   
        updates[param] = stepped_param      
    
    return updates

def sgd_updates(params, cost, learning_rate):
    """
            Using this function, specify how to update the parameters of the lbfgs_model as a dictionary
    """
    updates = OrderedDict({})
    grads = [T.grad(cost, p) for p in params]
    # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
    # same length, zip generates a list C of same size, where each element
    # is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3) , (a4, b4)]
    for param, grad in zip(params, grads):
        updates[param] = param - learning_rate * grad
    return updates

def set_values(name, param, pretrained):
    """
    Initialize a network parameter with pretrained values.
    We check that sizes are compatible.
    """
    param_value = param.get_value()
    if pretrained.size != param_value.size:
        raise Exception(
            "Size mismatch for parameter %s. Expected %i, found %i."
            % (name, param_value.size, pretrained.size)
        )
    param.set_value(np.reshape(
        pretrained, param_value.shape
    ).astype(np.float32))
