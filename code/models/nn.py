import theano
import theano.tensor as T
import numpy as np
from model_utils import shared, block_orth_normal_initializer
from collections import OrderedDict

class SparseLayer(object):
    def __init__(self, input_dim, set_zero=False, name='embedding_layer'):
        """
        Typically, input_dim is the vocabulary size,
        and output_dim the embedding dimension.
        """
        self.input_dim = input_dim
        self.name = name

        # Randomly generate weights
        value = np.zeros((input_dim, input_dim), dtype='float32')
        for i in range(input_dim):
            value[i,i] = 1.0
        if set_zero:
            value[0,0] = 0.0
        self.embeddings = theano.shared(value=value.astype(theano.config.floatX), name=self.name + '__embeddings')
        
    def link(self, input, with_batch=True):
        """
        Return the embeddings of the given indexes.
        Input: tensor of shape (dim*)
        Output: tensor of shape (dim*, output_dim)
        """
        self.input = input
        if with_batch:
            self.output = self.embeddings[self.input.flatten()].reshape((self.input.shape[0],self.input.shape[1],self.input_dim))
        else:
            self.output = self.embeddings[self.input]
        return self.output
    
    def tensor_link(self, input):
        self.input = input
        self.output = self.embeddings[self.input.flatten()].reshape((self.input.shape[0],self.input.shape[1],self.input.shape[2],self.input_dim))
        
        return self.output


class HiddenLayer(object):
    """
    Hidden layer with or without bias.
    Input: tensor of dimension (dims*, input_dim)
    Output: tensor of dimension (dims*, output_dim)
    """
    def __init__(self, input_dim, output_dim, bias=True, activation='sigmoid',
                 name='hidden_layer'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.name = name
        if activation is None:
            self.activation = None
        elif activation == 'tanh':
            self.activation = T.tanh
        elif activation == 'sigmoid':
            self.activation = T.nnet.sigmoid
        elif activation == 'softmax':
            self.activation = T.nnet.softmax
        elif activation == 'relu':
            self.activation = T.nnet.relu
        else:
            raise Exception("Unknown activation function: " % activation)

        # Initialize weights and bias
        self.weights = shared((input_dim, output_dim), name + '__weights')
        
        # Define parameters
        if self.bias:
            self.bias_weights = shared((output_dim,), name + '__bias')

            self.params = [self.weights, self.bias_weights]
            self.L2_sqr = (
                (self.weights ** 2).sum()
                + (self.bias_weights ** 2).sum()
            )
        else:
            self.params = [self.weights]
            self.L2_sqr = ( 
                (self.weights ** 2).sum()
            )


    def link(self, input):
        """
        The input has to be a tensor with the right
        most dimension equal to input_dim.
        """
        self.input = input
        self.linear_output = T.dot(self.input, self.weights)
        if self.bias:
            self.linear_output = self.linear_output + self.bias_weights
        if self.activation is None:
            self.output = self.linear_output
        else:
            self.output = self.activation(self.linear_output)
        return self.output


class EmbeddingLayer(object):
    """
    Embedding layer: word embeddings representations
    Input: tensor of dimension (dim*) with values in range(0, input_dim)
    Output: tensor of dimension (dim*, output_dim)
    """

    def __init__(self, input_dim, output_dim, name='embedding_layer'):
        """
        Typically, input_dim is the vocabulary size,
        and output_dim the embedding dimension.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name

        # Randomly generate weights
        self.embeddings = shared((input_dim, output_dim), self.name + '__embeddings')

        # Define parameters
        self.params = [self.embeddings]
        
        self.updates = OrderedDict({self.embeddings: self.embeddings / T.sqrt(T.sum(self.embeddings ** 2, axis=1, keepdims=True))})
        self.normalize = theano.function(inputs=[], outputs=[], updates=self.updates)
        
        self.L2_sqr = (
                (self.embeddings ** 2).sum()
            )
        
        self.zero_vec_tensor = T.vector()
        self.zero_vec = np.zeros(self.output_dim)
        self.set_word_zero = theano.function(
            [self.zero_vec_tensor], updates=[(self.embeddings, T.set_subtensor(self.embeddings[0,:], self.zero_vec_tensor))], allow_input_downcast=True)
         
        self.updates = OrderedDict({self.embeddings: self.embeddings / T.sqrt(T.sum(self.embeddings ** 2, axis=1)).dimshuffle(0,'x')})
        self.normalize = theano.function([], [], updates=self.updates)
         
    def set_dummy_zero(self):
        self.set_word_zero(self.zero_vec)  
    
    def reset_embeddings(self, wid2vec):
        new_weights = self.embeddings.get_value()  
        for wid in wid2vec:
            new_weights[wid,:] = wid2vec[wid]
        self.embeddings.set_value(new_weights)
        
    def link(self, input, with_batch=True):
        """
        Return the embeddings of the given indexes.
        Input: tensor of shape (dim*)
        Output: tensor of shape (dim*, output_dim)
        """
        self.input = input
        if with_batch:
            self.output = self.embeddings[self.input.flatten()].reshape((self.input.shape[0],self.input.shape[1],self.output_dim))
        else:
            self.output = self.embeddings[self.input]
        return self.output

    def tensor_link(self, input):
        self.input = input
        self.output = self.embeddings[self.input.flatten()].reshape((self.input.shape[0],self.input.shape[1],self.input.shape[2],self.output_dim))
        
        return self.output
    
class DropoutLayer(object):
    """
    Dropout layer. Randomly set to 0 values of the input
    with probability p.
    """
    def __init__(self, p, name, is_train):
        """
        p has to be between 0 and 1 (1 excluded).
        p is the probability of dropping out a unit, so
        setting p to 0 is equivalent to have an identity layer.
        """
        assert 0. <= p < 1.
        self.p = p
        self.rng = T.shared_randomstreams.RandomStreams(seed=123456)
        self.name = name
        self.is_train = is_train
    
    def generate_fix_mask(self, shape):
        self.fix_mask = self.rng.binomial(n=1, p=1-self.p, size=shape,
                                     dtype=theano.config.floatX)
        
    def link(self, input):
        """
        Dropout link: we just apply mask to the input.
        """
        if self.p > 0:
            mask = self.rng.binomial(n=1, p=1-self.p, size=input.shape,
                                     dtype=theano.config.floatX)
            self.output = input * mask
        else:
            self.output = input

        return self.output
    
    def apply_dropout(self, inputs):
        input_train = self.link(inputs)
        input_test = (1 - self.p) * inputs
        inputs = T.switch(T.neq(self.is_train, 0), input_train, input_test)
        return inputs

class FastLSTM(object):
    """
    LSTM with faster implementation (supposedly).
    Not as expressive as the previous one though, because it doesn't include the peepholes connections.
    """
    def __init__(self, input_dim, hidden_dim, name):
        """
        Initialize neural network.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.name = name

        self.W = shared((input_dim, hidden_dim * 4), name + 'W')
        self.U = shared((hidden_dim, hidden_dim * 4), name + 'U')
        #self.W = block_orth_normal_initializer((input_dim, hidden_dim * 4), [input_dim,], [hidden_dim] * 4, name + 'W')
        #self.U = block_orth_normal_initializer((hidden_dim, hidden_dim * 4), [hidden_dim,], [hidden_dim] * 4, name + 'U') 
        
        self.b = shared((hidden_dim * 4, ), name + 'b')

        self.c_0 = shared((hidden_dim,), name + '__c_0')
        self.h_0 = shared((hidden_dim,), name + '__h_0')

        self.params = [self.W, self.U, self.b]
        
    def link(self, input, with_batch=True):
        """
        Propagate the input through the network and return the last hidden vector.
        The whole sequence is also accessible through self.h
        """
        def split(x, n, dim):
            if x.ndim == 1:
                return x[n*dim:(n+1)*dim]
            else:
                return x[:, n*dim:(n+1)*dim]

        def recurrence(x_t, c_tm1, h_tm1):
            p = x_t + T.dot(h_tm1, self.U)
            i = T.nnet.sigmoid(split(p, 0, self.hidden_dim))
            f = T.nnet.sigmoid(split(p, 1, self.hidden_dim) + 1.0)
            o = T.nnet.sigmoid(split(p, 2, self.hidden_dim))
            c = T.tanh(split(p, 3, self.hidden_dim))
            c = f * c_tm1 + i * c
            h = o * T.tanh(c)
            return c, h
        
        if with_batch:
            # input: seq, batch, input_dim
            self.input = input.dimshuffle(1, 0, 2) # seq, batch, input
            preact = T.dot(self.input, self.W) + self.b
            outputs_info = [T.alloc(x, self.input.shape[1], self.hidden_dim) for x in [self.c_0, self.h_0]]
            
        else:
            self.input = input # seq, input_dim
            preact = T.dot(self.input, self.W) + self.b # seq, 4*hidden_dim
            outputs_info = [self.c_0, self.h_0]
        
        [_, h], _ = theano.scan(
            fn=recurrence,
            sequences=preact,
            outputs_info=outputs_info,
            n_steps=self.input.shape[0]
        )
        self.h = h
        self.output = h[-1]

        return h    
    
    def mask_link(self, input, mask):
        """
        Propagate the input through the network and return the last hidden vector.
        The whole sequence is also accessible through self.h
        """
        def split(x, n, dim):
            if x.ndim == 1:
                return x[n*dim:(n+1)*dim]
            else:
                return x[:, n*dim:(n+1)*dim]

        def recurrence(x_t, m_, c_tm1, h_tm1):
            p = x_t + T.dot(h_tm1, self.U)
            i = T.nnet.sigmoid(split(p, 0, self.hidden_dim))
            f = T.nnet.sigmoid(split(p, 1, self.hidden_dim) + 1.0)
            o = T.nnet.sigmoid(split(p, 2, self.hidden_dim))
            c = T.tanh(split(p, 3, self.hidden_dim))
            c = f * c_tm1 + i * c
            h = o * T.tanh(c)
            h = m_[:, None] * h + (1. - m_)[:, None] * h_tm1

            return c, h
        
        # input: seq, batch, input_dim
        self.input = input.dimshuffle(1, 0, 2) # seq, batch, input
        self.mask = mask.dimshuffle(1, 0)  # seq, batch
        
        preact = T.dot(self.input, self.W) + self.b
        outputs_info = [T.alloc(x, self.input.shape[1], self.hidden_dim) for x in [self.c_0, self.h_0]]

        [_, h], _ = theano.scan(
            fn=recurrence,
            sequences=[preact, self.mask],
            outputs_info=outputs_info,
            n_steps=self.input.shape[0]
        )
        self.h = h
        self.output = h[-1]

        return h    
             
class KBLSTM(object):
    def __init__(self, input_dim, hidden_dim, name='FastLSTM'):
        """
        Initialize neural network.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.name = name

        self.W = shared((input_dim, hidden_dim * 5), name + 'W')
        self.U = shared((hidden_dim, hidden_dim * 5), name + 'U')
        #self.W = block_orth_normal_initializer((input_dim, hidden_dim * 5), [input_dim,], [hidden_dim] * 5, name + 'W')
        #self.U = block_orth_normal_initializer((hidden_dim, hidden_dim * 5), [hidden_dim,], [hidden_dim] * 5, name + 'U') 
        
        self.b = shared((hidden_dim * 5, ), name + 'b')

        self.c_0 = shared((hidden_dim,), name + '__c_0')
        self.h_0 = shared((hidden_dim,), name + '__h_0')
        self.s_0 = shared((hidden_dim,), name + '__s_0')

        self.params = [self.W, self.U, self.b]

    def link(self, input, with_batch=True):
        """
        Propagate the input through the network and return the last hidden vector.
        The whole sequence is also accessible through self.h
        """
        def split(x, n, dim):
            if x.ndim == 1:
                return x[n*dim:(n+1)*dim]
            else:
                return x[:, n*dim:(n+1)*dim]
            
        def recurrence(x_t, c_tm1, h_tm1, s_tm1):
            p = x_t + T.dot(h_tm1, self.U)
            i = T.nnet.sigmoid(split(p, 0, self.hidden_dim))
            f = T.nnet.sigmoid(split(p, 1, self.hidden_dim) + 1.0)
            o = T.nnet.sigmoid(split(p, 2, self.hidden_dim))
            c = T.tanh(split(p, 3, self.hidden_dim))
            c = f * c_tm1 + i * c
            h = o * T.tanh(c)
        
            g = T.nnet.sigmoid(split(p, 4, self.hidden_dim))
            s = g * T.tanh(c) 
            
            return c, h, s
        
        self.input = input.dimshuffle(1, 0, 2) # seq, batch, input

        preact = T.dot(self.input, self.W) + self.b
        outputs_info = [T.alloc(x, self.input.shape[1], self.hidden_dim) for x in [self.c_0, self.h_0, self.s_0]]
        
        [_, h, s], _ = theano.scan(
            fn=recurrence,
            sequences=[preact],
            outputs_info=outputs_info,
            n_steps=self.input.shape[0]
        )
        self.h = h
        self.s = s
        self.output = h[-1]

        return self.output    

'''               
def log_sum_exp(x, axis=None):
    """
    Sum probabilities in the log-space.
    """
    xmax = x.max(axis=axis, keepdims=True)
    xmax_ = x.max(axis=axis)
    return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))
'''
    
def logsumexp(x, axis):
    """
    :param x: 1D: batch, 2D: n_y, 3D: n_y
    :return:
    """
    x_max = T.max(x, axis=axis, keepdims=True)
    return T.log(T.sum(T.exp(x - x_max), axis=axis, keepdims=True)) + x_max

class CRFLayer(object):
    def __init__(self, n_tags, name='CRF'):
        """
        Initialize neural network.
        """
        self.name = name
        
        self.transitions = shared((n_tags, n_tags), name + '__transitions')
        self.params = [self.transitions]
            
    def forward_step(self, obs_t, y_t, y_prev, obs_score_prev, z_scores_prev, trans, batch_size):
        obs_score_t = obs_score_prev + trans[y_t, y_prev] + obs_t[T.arange(batch_size), y_t]  # 1D: Batch
        z_sum = z_scores_prev.dimshuffle(0, 'x', 1) + trans  # 1D: Batch, 2D: n_y, 3D: n_y
        z_scores_t = logsumexp(z_sum, axis=2).reshape(obs_t.shape) + obs_t  # 1D: Batch, 2D: n_y
        return y_t, obs_score_t, z_scores_t


    def y_prob(self, observations, y, batch_size):
        """
        observations: batch, seqlen, Y
        y: batch, seqlen
        return: batch,
        """
        self.input = observations.dimshuffle(1, 0, 2) # seqlen, batch, Y
        self.y = y.dimshuffle(1, 0) # seqlen, batch
        
        y_score0 = self.input[0][T.arange(batch_size), self.y[0]]  # 1D: Batch
        z_score0 = self.input[0]  # 1D: Batch, 2D: n_y

        [_, y_scores, z_scores], _ = theano.scan(fn=self.forward_step,
                                                 sequences=[self.input[1:], self.y[1:]],
                                                 outputs_info=[self.y[0], y_score0, z_score0],
                                                 non_sequences=[self.transitions, batch_size])

        y_score = y_scores[-1]
        z_score = logsumexp(z_scores[-1], axis=1).flatten()

        return  z_score - y_score

    def viterbi_backward(self, nodes_t, max_node_t, batch_size):
        return nodes_t[T.arange(batch_size), max_node_t]
    
    def viterbi_forward(self, obs_t, score_prev, trans):
        score = score_prev.dimshuffle(0, 'x', 1) + trans + obs_t.dimshuffle(0, 1, 'x')
        max_scores_t, max_nodes_t = T.max_and_argmax(score, axis=2)
        return max_scores_t, T.cast(max_nodes_t, dtype='int32')
    
    def viterbi(self, observations, batch_size):
        """
        observations: batch, seqlen, Y
        return: batch, seqlen
        """
        self.input = observations.dimshuffle(1, 0, 2) # seqlen, batch, Y
        scores0 = self.input[0]
        [max_scores, max_nodes], _ = theano.scan(fn=self.viterbi_forward,
                                                 sequences=[self.input[1:]],
                                                 outputs_info=[scores0, None],
                                                 non_sequences=self.transitions)
        max_last_node = T.cast(T.argmax(max_scores[-1], axis=1), dtype='int32') # seqlen-1

        nodes, _ = theano.scan(fn=self.viterbi_backward,
                               sequences=max_nodes[::-1],
                               outputs_info=max_last_node,
                               non_sequences=batch_size)

        return T.concatenate([nodes[::-1].dimshuffle(1, 0), max_last_node.dimshuffle((0, 'x'))], 1)



