'''
Created on Dec 11, 2016

@author: bishan
'''

import os
import scipy.io
import theano
import theano.tensor as T
import cPickle

from nn import *
from optimization import Optimization 
from model_utils import set_values

class SeqModel(object):
    """
    Network architecture.
    """
    def __init__(self):
        """
        Initialize the model. We either provide the parameters and a path where
        we store the models, or the location of a trained model.
        """
        self.components = {}
    
    def init_params(self, args):
        self.model_path = args.model_path        
        self.mappings_path = self.model_path + 'mappings.pkl'
        
        self.word_dim = args.word_dim
        self.hidden_dim = args.hidden_dim
        self.dropout = args.dropout
        self.lstm_bidirect = args.lstm_bidirect
        self.lr_method = args.lr_method
        self.crf = args.crf
        self.concept_dim = args.concept_dim
        
        # Save the parameters to disk
        with open(self.model_path + 'parameters.pkl', 'wb') as f:
            cPickle.dump(vars(args), f)
        
    def load_params(self, model_path):
        self.model_path = model_path
        self.mappings_path = self.model_path + 'mappings.pkl'
        
        f = open(self.model_path + 'parameters.pkl', 'rb')
        parameters = cPickle.load(f)
        f.close()
        
        self.word_dim = parameters['word_dim']
        self.hidden_dim = parameters['hidden_dim']
        self.dropout = parameters['dropout']
        self.lstm_bidirect = parameters['lstm_bidirect']
        self.lr_method = parameters['lr_method']
        self.crf = parameters['crf']
        self.concept_dim = parameters['concept_dim']

    def save_mappings(self, id_to_word, id_to_concept, id_to_tag):
        """
        We need to save the mappings if we want to use the model later.
        """
        self.id_to_word = id_to_word
        self.id_to_tag = id_to_tag
        self.id_to_concept = id_to_concept
        with open(self.mappings_path, 'wb') as f:
            mappings = {
                'id_to_word': self.id_to_word,
                'id_to_concept': self.id_to_concept,
                'id_to_tag': self.id_to_tag
            }
            cPickle.dump(mappings, f)

    def reload_mappings(self):
        """
        Load mappings from disk.
        """
        with open(self.mappings_path, 'rb') as f:
            mappings = cPickle.load(f)
        self.id_to_word = mappings['id_to_word']
        self.id_to_concept = mappings['id_to_concept']
        self.id_to_tag = mappings['id_to_tag']
        
    def add_component(self, param):
        """
        Add a new parameter to the network.
        """
        if param.name in self.components:
            raise Exception('The network already has a parameter "%s"!'
                            % param.name)
        self.components[param.name] = param
           
    def save(self, epoch=-1):
        """
        Write components values to disk.
        """
        for name, param in self.components.items():
            if epoch > 0:
                param_path = self.model_path + "%s_%d.mat" % (name, epoch)
                param_path = self.model_path + "%s_best.mat" % name
            else:
                param_path = self.model_path + "%s_final.mat" % name
            if hasattr(param, 'params'):
                param_values = {p.name: p.get_value() for p in param.params}
            else:
                param_values = {name: param.get_value()}
            scipy.io.savemat(param_path, param_values)

    def reload(self, epoch=-1):
        """
        Load components values from disk.
        """
        for name, param in self.components.items():
            if epoch > 0:
                param_path = self.model_path + "%s_best.mat" % name
            else:
                param_path = self.model_path + "%s_final.mat" % name
            if not os.path.exists(param_path):
                continue
            param_values = scipy.io.loadmat(param_path)
            if hasattr(param, 'params'):
                for p in param.params:
                    if p.name in param_values:
                        set_values(p.name, p, param_values[p.name])
            else:
                set_values(name, param, param_values[name])
            
    def build(self):
        """
        Build the network.
        """
        # Training parameters
        n_words = len(self.id_to_word)
        n_concepts = len(self.id_to_concept)
        n_tags = len(self.id_to_tag)     
            
        # Network variables
        self.is_train = T.iscalar('is_train')
        self.word_ids = T.imatrix(name='word_ids')
        
        self.concept_ids = T.itensor3(name='concept_ids')
        self.kb_mask = T.ftensor3(name='kb_mask')
        self.y = T.imatrix(name='y') # batch, seqlen
         
        #
        # Word inputs
        #
        self.word_layer = EmbeddingLayer(n_words, self.word_dim, name='word_layer')
        word_input = self.word_layer.link(self.word_ids) # batch, seqlen, word_dim
    
        if self.dropout:
            dropout_layer = DropoutLayer(self.dropout, "dropout_layer", self.is_train)
            word_input = dropout_layer.apply_dropout(word_input)
                
        inputs = []
        input_dim = self.word_dim
        inputs.append(word_input)
        
        inputs = T.concatenate(inputs, axis=2)
        
        word_lstm_for = KBLSTM(input_dim, self.hidden_dim, name='word_lstm_for')
        word_lstm_for.link(inputs)
        word_for_output = word_lstm_for.h.dimshuffle(1, 0, 2) # batch, seq, hidden_dim             
        word_for_output_s = word_lstm_for.s.dimshuffle(1, 0, 2) # batch, seq, hidden_dim             

        word_lstm_rev = KBLSTM(input_dim, self.hidden_dim, name='word_lstm_rev')
        word_lstm_rev.link(inputs[:, ::-1, :]) # batch, seq_rev, input_dim
        word_rev_output = word_lstm_rev.h.dimshuffle(1, 0, 2)[:, ::-1, :]
        word_rev_output_s = word_lstm_rev.s.dimshuffle(1, 0, 2)[:, ::-1, :]
        
        layer_output = T.concatenate([word_for_output, word_rev_output], axis=2)
        layer_output_s = T.concatenate([word_for_output_s, word_rev_output_s], axis=2)
            
        # concept layer
        self.concept_layer = EmbeddingLayer(n_concepts, self.concept_dim, name='concept_layer')
        concept_output = self.concept_layer.tensor_link(self.concept_ids) # batch, seqlen, concept_num, concept_dim
        
        label_dim = self.hidden_dim*2
        bilinear_layer = HiddenLayer(label_dim, self.concept_dim, bias=False, name='bilinear_layer', activation=None)
        layer_output_1 = bilinear_layer.link(layer_output) # batch, seqlen, word_feature_dim
        word_output = T.extra_ops.repeat(layer_output_1, self.concept_ids.shape[2], axis=2)
        word_output = word_output.reshape((layer_output_1.shape[0], layer_output_1.shape[1], layer_output_1.shape[2], self.concept_ids.shape[2]))
        word_output = word_output.dimshuffle(0,1,3,2) # batch, seqlen, concept_num, word_feature_dim
        alpha = T.sum(word_output * concept_output, axis=3) + T.log(self.kb_mask)
        
        beta_layer = HiddenLayer(label_dim, label_dim, bias=False, name='beta_layer', activation=None)
        layer_output_2 = beta_layer.link(layer_output) # batch, seqlen, label_dim
        beta = T.sum(layer_output_2 * layer_output_s, axis=2, keepdims=True) # batch, seqlen, 1
        
        log_sum = logsumexp(T.concatenate([alpha, beta], axis=2), axis=2)
        alpha = T.exp(alpha - log_sum)
        beta = T.exp(beta - log_sum)
        
        knowledge_output = T.sum(alpha.dimshuffle(0,1,2,'x') * concept_output, axis=2) 
        mix_output = knowledge_output + beta * layer_output_s # concept_dim
        
        final_output = layer_output + mix_output
         
        input_dim = self.hidden_dim*2
        final_layer = HiddenLayer(input_dim, n_tags, bias=False, name='final_layer', activation=None)
        final_output = final_layer.link(final_output) # batch_size, seq_size, Y  
    
        if self.crf:
            crf_layer = CRFLayer(n_tags, name='crf_layer')
            costs = crf_layer.y_prob(final_output, self.y, final_output.shape[0])
            self.cost = T.mean(costs) #+ l1
            self.y_pred = crf_layer.viterbi(final_output, final_output.shape[0]) # seqlen, batch
        else:
            final_output_flat = final_output.reshape((final_output.shape[0]*final_output.shape[1], final_output.shape[2]))
            p_y_given_x = T.nnet.softmax(final_output_flat) # batch_size*seq_size, Y   
            y_flat = self.y.flatten()
            self.cost = -T.mean(T.log(p_y_given_x)[T.arange(y_flat.shape[0]), y_flat]) #+ l1# (batch*len, 1) 
            y_pred_flat = T.argmax(p_y_given_x, axis=1)
            self.y_pred = y_pred_flat.reshape((self.y.shape[0], self.y.shape[1]))        
        
        self.cost += 1e-05 * (bilinear_layer.L2_sqr + beta_layer.L2_sqr + final_layer.L2_sqr)
           
        # Network parameters
        self.params = []
    
        self.add_component(self.word_layer)
        self.params.extend(self.word_layer.params)
         
        self.add_component(self.concept_layer)

        self.add_component(word_lstm_for)
        self.params.extend(word_lstm_for.params)
        self.add_component(word_lstm_rev)
        self.params.extend(word_lstm_rev.params)
        
        self.add_component(bilinear_layer)
        self.params.extend(bilinear_layer.params)
        
        self.add_component(beta_layer)
        self.params.extend(beta_layer.params)
           
        if self.crf:
            self.add_component(crf_layer)
            self.params.extend(crf_layer.params)
            
        self.add_component(final_layer)
        self.params.extend(final_layer.params)
        
    def get_train_function(self, train_x, train_y): 
        train_set_x = []
        for i in range(len(train_x)):
            train_set_x.append(theano.shared(np.asarray(train_x[i], dtype=theano.config.floatX), borrow=True))
        train_set_y = theano.shared(np.asarray(train_y, dtype=theano.config.floatX), borrow=True)
        train_set_y = T.cast(train_set_y, 'int32')
        
        # Parse optimization method parameters
        if "-" in self.lr_method:
            lr_method_name = self.lr_method[:self.lr_method.find('-')]
            lr_method_parameters = {}
            for x in self.lr_method[self.lr_method.find('-') + 1:].split('-'):
                split = x.split('_')
                assert len(split) == 2
                lr_method_parameters[split[0]] = float(split[1])
        else:
            lr_method_name = self.lr_method
            lr_method_parameters = {}
            
        seqlen = T.iscalar()
        indices = T.ivector()
        updates = Optimization(clip=5.0).get_updates(lr_method_name, self.cost, self.params, **lr_method_parameters)
    
        f_train = theano.function(
            inputs = [indices, seqlen], 
            outputs = self.cost, 
            updates = updates,
            on_unused_input='warn',
            givens={
                self.word_ids: T.cast(train_set_x[0][indices][:,:seqlen], 'int32'),
                self.concept_ids: T.cast(train_set_x[1][indices][:,:seqlen, :], 'int32'),
                self.kb_mask: train_set_x[2][indices][:,:seqlen, :],
                self.y: train_set_y[indices][:,:seqlen],
                self.is_train: np.cast['int32'](1)
                }
        )      
        return f_train   
    
    def get_eval_function(self, dev_x, dev_y): 
        dev_set_x = []
        for i in range(len(dev_x)):
            dev_set_x.append(theano.shared(np.asarray(dev_x[i], dtype=theano.config.floatX), borrow=True))
        dev_set_y = theano.shared(np.asarray(dev_y, dtype=theano.config.floatX), borrow=True)
        dev_set_y = T.cast(dev_set_y, 'int32')
    
        seqlen = T.iscalar()
        indices = T.ivector()
        
        f_eval = theano.function(
            inputs=[indices, seqlen],
            outputs=self.y_pred,
            on_unused_input='warn',
            givens={
                self.word_ids: T.cast(dev_set_x[0][indices][:,:seqlen], 'int32'),
                self.concept_ids: T.cast(dev_set_x[1][indices][:,:seqlen, :], 'int32'),
                self.kb_mask: dev_set_x[2][indices][:,:seqlen, :],
                self.y: dev_set_y[indices][:,:seqlen],
                self.is_train: np.cast['int32'](0)
            }
        )
        
        return f_eval     
    

