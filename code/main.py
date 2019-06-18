'''
Created on Dec 11, 2016

@author: bishan
'''
import argparse

from models.loader import *
from models.model import *
from models.eval import *
from collections import defaultdict
#import theano.tensor as T
import time
from utils import *

rng = np.random.RandomState(3435)

parser = argparse.ArgumentParser(description='KBLSTM model')
parser.add_argument('--word-dim', type=int, default=300,
                    help='word embedding dimension')
parser.add_argument('--hidden-dim', type=int, default=50,
                    help='LSTM hidden layer size')
parser.add_argument('--batch-size', type=int, default=1,
                    help='batch size for training')
parser.add_argument("--num-epochs", type=int, default=5,
                     help="number of training epochs")
parser.add_argument("--lstm-bidirect", type=int, default=1,
                    help="Use a bidirectional LSTM")
parser.add_argument("--crf", type=int, default=1,
                    help="Use CRF (0 to disable)")
parser.add_argument("--dropout", type=float, default=0.5,
                     help="Droupout on the input (0 = no dropout)")
parser.add_argument("--lr-method", type=str, default="adam",
                     help="Learning method (SGD, Adadelta, Adam..)")

parser.add_argument("--embedding-dir", type=str, default="",
                    help="location of embedding files")
parser.add_argument("--embedding-dim", type=int, default=300,
                    help="dimension of pre-trained word embeddings")
parser.add_argument("--embedding-file", type=str, default="paragram_300_sl999.txt",
                    help="pre-trained embedding file")
parser.add_argument("--concept-dim", type=int, default=100,
                     help="concept embedding dimension")
parser.add_argument("--concept-embedding", type=str, default="wn_concept2vec.txt",
                     help="concept embedding file")

parser.add_argument("--data-dir", type=str, default="",
                     help="location of data")
parser.add_argument("--train-file", type=str, default="train.txt",
                     help="location of training data")
parser.add_argument("--dev-file", type=str, default="dev.txt",
                     help="location of validation data")
parser.add_argument("--test-file", type=str, default="test.txt",
                     help="location of test data")
parser.add_argument("--model-path", type=str, default="",
                     help="location of the model parameters")

def load_sentences(datafile): 
    sentences = []
    sentence = []
    for line in codecs.open(datafile, 'r', 'utf-8'):
        if line == '\n':
            sentences.append(sentence)
            sentence = []
        else:
            fields = line.rstrip().split('\t')
            sentence.append(fields)
    if len(sentence) > 0:
        sentences.append(sentence)
        
    return sentences

def build_word_dict(sentences):
    word_dict = defaultdict(int)
    for s in sentences:
        for fields in s:
            word_dict[fields[0]] += 1
    return word_dict

def build_concept_dict(sentences):
    concept_dict = defaultdict(int)
    for s in sentences:
        for fields in s:
            for c in fields[-2].split(','):
                if c != "O":
                    concept_dict[c] += 1

    return concept_dict

def build_tag_dict(sentences):
    tag_dict = defaultdict(int)
    for s in sentences:
        for fields in s:
            tag_dict[fields[-1]] += 1
    return tag_dict

def prepare_dataset(sentences, word_to_id, concept_to_id, tag_to_id, isTrain):            
    max_concept_len = max([len(w[-2].split(',')) for s in sentences for w in s])
    max_len = max([len(s) for s in sentences])
    
    x_word = np.zeros((len(sentences), max_len)).astype('int32')
    x_kb = np.zeros((len(sentences), max_len, max_concept_len)).astype('int32')
    x_kb_mask = np.zeros((len(sentences), max_len, max_concept_len))
    
    y = np.zeros((len(sentences), max_len)).astype('int32')
    
    indices = []
    for idx, s in enumerate(sentences):
        indices.append((idx, len(s)))
        for sid, x in enumerate(s):
            x_word[idx, sid] = word_to_id[x[0]] if x[0] in word_to_id else 0
            
            k = 0
            for concept in x[-2].split(','):
                if concept in concept_to_id:
                    x_kb[idx, sid, k] = concept_to_id[concept] #if concept in kb_to_id else 0
                    x_kb_mask[idx, sid, k] = 1.0
                    k += 1
    
        if isTrain:
            y[idx, :len(s)] = [tag_to_id[w[-1]] for w in s]
        else:
            y[idx, :len(s)] = [tag_to_id[w[-1]] if w[-1] in tag_to_id else tag_to_id["O"] for w in s]   
        
    return [x_word, x_kb, x_kb_mask], y, indices

def shuffle_batches(sentences, batch_size, isTrain):
    if isTrain:
        rng.shuffle(sentences)
    sentences.sort(key=lambda s: s[1])  # sort with n_words
    
    batches_index = []
    batch = []
    batches_seqlen = []
    prev_n_words = sentences[0][1]
    for s in sentences:
        n_words = s[1]
        if len(batch) == batch_size or prev_n_words != n_words:
            batches_index.append(batch)
            batches_seqlen.append(prev_n_words)
            batch = []
        batch.append(s[0]) 
        prev_n_words = n_words
        
    if len(batch) > 0:
        batches_index.append(batch)
        batches_seqlen.append(prev_n_words)
    return batches_index, batches_seqlen
    
def evaluate_acc(gold_y, pred_y):
    rec_num = ((gold_y == pred_y) & (pred_y != 0)).sum()
    rec_den = (gold_y != 0).sum()
    prec_num = ((gold_y == pred_y) & (pred_y != 0)).sum()
    prec_den = (pred_y != 0).sum()
    micro_prec = 0.0 if prec_den == 0.0 else float(prec_num)/prec_den
    micro_rec = 0.0 if rec_den == 0.0 else float(rec_num)/rec_den
    micro_f1 = 0.0 if micro_prec+micro_rec == 0.0 else float(2*micro_prec*micro_rec)/(micro_prec+micro_rec)
    print "Micro score ", round(micro_prec*100,1), round(micro_rec*100,1), round(micro_f1*100,1), prec_num, prec_den, rec_num, rec_den
    return micro_f1

def gen_word_embeddings(word_to_id, embedding_dim, embedding_file):                
    print 'Loading pretrained embeddings from %s...' % embedding_file
    wid2vec = {}
    with open(embedding_file, 'r') as rf:
        for line in rf.readlines():
            sp = line.split(' ')
            assert len(sp) == embedding_dim + 1
            wordstr = sp[0]
            if wordstr in word_to_id:
                wid = word_to_id[wordstr]
                wid2vec[wid] = np.asarray([float(x) for x in sp[1:]], 'float32')
    print ('%i / %i (%.2f%%) words have been initialized with pretrained embeddings.') % (len(wid2vec), len(word_to_id), 100. * (len(wid2vec)) / len(word_to_id))      
    return wid2vec

def gen_concept_embeddings(word_to_id, embedding_dim, embedding_file):                
    print 'Loading pretrained concept embeddings from %s...' % embedding_file
    wid2vec = {}
    with open(embedding_file, 'r') as rf:
        for line in rf.readlines():
            sp = line.split(' ')
            assert len(sp) == embedding_dim + 1
            wordstr = sp[0]
            if wordstr in word_to_id:
                word = word_to_id[wordstr]
                wid2vec[word] = [float(x) for x in sp[1:]]  
    wid2vec[0] = [0 for _ in range(embedding_dim)] 
    print ('%i / %i (%.2f%%) concepts have been initialized with pretrained embeddings.') % (len(wid2vec), len(word_to_id), 100. * (len(wid2vec)) / len(word_to_id))      
    return wid2vec

def train():
    train_sentences = load_sentences(args.train_file)
    dev_sentences = load_sentences(args.dev_file)
    
    word_dict = build_word_dict(train_sentences+dev_sentences)
    tag_dict = build_tag_dict(train_sentences+dev_sentences)
    concept_dict = build_concept_dict(train_sentences+dev_sentences)
    
    dico_words_train = word_dict.copy()
   
    word_to_id, id_to_word = create_mapping_with_unk(word_dict)
    concept_to_id, id_to_concept = create_mapping_with_unk(concept_dict)
    tag_to_id, id_to_tag = create_mapping(tag_dict)
    
    singletons = set([word_to_id[k] for k, v in dico_words_train.items() if v == 1])
    p = 0.01
    for s in train_sentences:
        for x in s:
            words = x[0].split(' ')
            for w in words:
                if w in singletons and rng.uniform() < p:
                    w = "<unk>"
    
    train_x, train_y, train_sentences = prepare_dataset(train_sentences, word_to_id, concept_to_id, tag_to_id, True)
    dev_x, dev_y, dev_sentences = prepare_dataset(dev_sentences, word_to_id, concept_to_id, tag_to_id, False)
    
    model = SeqModel() 
    model.init_params(args)
    model.save_mappings(id_to_word, id_to_concept, id_to_tag)
    
    # Build the model
    model.build()
    
    if args.embedding_file:
        word2vec = gen_word_embeddings(word_to_id, args.embedding_dim, args.embedding_file)
        model.word_layer.reset_embeddings(word2vec)

    concept2vec = gen_concept_embeddings(concept_to_id, args.concept_dim, args.concept_embedding) 
    model.concept_layer.reset_embeddings(concept2vec)
     
    f_train = model.get_train_function(train_x, train_y)
    f_eval = model.get_eval_function(dev_x, dev_y)
     
    print 'start training'
    dev_batch_indices, dev_batch_seqlens = shuffle_batches(dev_sentences, args.batch_size, False)
    
    train_batch_indices, train_batch_seqlens = shuffle_batches(train_sentences, args.batch_size, True)
    batch_indices = range(len(train_batch_indices))
        
    best_score = 0.0
    epoch = 0 
    while (epoch < args.num_epochs):
        total_cost = 0  
        start_time = time.time()
        epoch = epoch + 1
    
        for minibatch_index in range(len(batch_indices)):
            j = batch_indices[minibatch_index]
            indices = train_batch_indices[j]
            batch_width = train_batch_seqlens[j]
        
            cost = f_train(indices, batch_width)
            total_cost += cost

        print('epoch: %i, training time: %.2f secs, train cost: %f' % (epoch, time.time()-start_time, total_cost))
        
        gold_y = []
        pred_y = []
        for minibatch_index in range(len(dev_batch_indices)):
            indices = dev_batch_indices[minibatch_index]
            batch_width = dev_batch_seqlens[minibatch_index]
            out = f_eval(indices, batch_width)
            y_gold_batch = dev_y[indices][:,:batch_width]
            y_pred_batch = out
            gold_y.extend(list(y_gold_batch.flatten()))
            pred_y.extend(list(y_pred_batch.flatten()))
        
        gold_y = np.asarray(gold_y, dtype='int32')
        pred_y = np.asarray(pred_y, dtype='int32')
    
        eval_score = evaluate_acc(gold_y, pred_y)
        if eval_score > best_score:
            model.save(epoch)
            best_score = eval_score
            
    model.save()

def inference():
    model = SeqModel()
    model.load_params(args.model_path)
    model.reload_mappings()
    
    word_to_id = {v:k for k,v in model.id_to_word.items()}
    concept_to_id = {v:k for k,v in model.id_to_concept.items()}
    tag_to_id = {v:k for k,v in model.id_to_tag.items()}
    
    save_word_idx_map(concept_to_id, os.path.join(args.data_dir, "concept_to_id.txt"))
    
    test_sentences = load_sentences(args.test_file)
    test_x, test_y, test_indices = prepare_dataset(test_sentences, word_to_id, concept_to_id, tag_to_id, False)
    
    # Build the model
    model.build()
    # Reload previous model
    model.reload(1)
    
    f_eval = model.get_eval_function(test_x, test_y)
    
    gold_y = []
    pred_y = []
    test_batch_indices, test_batch_seqlens = shuffle_batches(test_indices, args.batch_size, False) 
    for minibatch_index in range(len(test_batch_indices)):
        indices = test_batch_indices[minibatch_index]
        batch_width = test_batch_seqlens[minibatch_index]

        y_gold_batch = test_y[indices][:,:batch_width]
        out = f_eval(indices, batch_width)
        y_pred_batch = out
        gold_y.extend(list(y_gold_batch.flatten()))
        pred_y.extend(list(y_pred_batch.flatten()))
    
    gold_y = np.asarray(gold_y, dtype='int32')
    pred_y = np.asarray(pred_y, dtype='int32')
    
    evaluate_acc(gold_y, pred_y)
   
def main():
    if os.path.isfile(args.train_file):
        train()
    if os.path.isfile(args.test_file):
        inference()
    
if __name__ == '__main__':
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
        
    if args.data_dir:
        args.train_file = os.path.join(args.data_dir, args.train_file)
        args.dev_file = os.path.join(args.data_dir, args.dev_file)
        args.test_file = os.path.join(args.data_dir, args.test_file)
    if args.embedding_dir:
        args.embedding_file = os.path.join(args.embedding_dir, args.embedding_file)
        args.concept_embedding = os.path.join(args.embedding_dir, args.concept_embedding)
    
    main()
    