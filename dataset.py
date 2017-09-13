from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import time
import sys
import os
import re
import operator
import gensim

from random import shuffle
from utils import *


def get_ethnicity_data(data_dir, params):
    is_ethnicity = params['ethnicity']

    for root, dir, files in os.walk(data_dir):
        unigram_set = []
        bigram_set = []
        trigram_set = []
        length_set = []
        labels = []

        unigram2idx = {}
        idx2unigram = {}
        bigram2idx = {}
        idx2bigram = {}
        trigram2idx = {}
        idx2trigram = {}
        country2idx = {}
        idx2country = {}
        country2ethnicity = {}
        name_max_len = 0

        train_set = []
        valid_set = []
        test_set = []

        for file_cnt, file_name in enumerate(sorted(files)):
            data = open(os.path.join(root, file_name))
            file_len = 0
            
            if file_name == '0_unigram_to_idx.txt':
                for k, line in enumerate(data):
                    file_len = k + 1
                    unigram, index = line[:-1].split('\t')
                    unigram2idx[unigram] = int(index)
                    idx2unigram[int(index)] = unigram
            elif file_name == '1_bigram_to_idx.txt':
                for k, line in enumerate(data):
                    file_len = k + 1
                    bigram, index = line[:-1].split('\t')
                    bigram2idx[bigram] = int(index)
                    idx2bigram[int(index)] = bigram
            elif file_name == '2_trigram_to_idx.txt':
                for k, line in enumerate(data):
                    file_len = k + 1
                    trigram, index = line[:-1].split('\t')
                    trigram2idx[trigram] = int(index)
                    idx2trigram[int(index)] = trigram
            elif file_name == 'country_to_idx.txt':
                for k, line in enumerate(data):
                    file_len = k + 1
                    country, index = line[:-1].split('\t')
                    if not is_ethnicity:
                        index = k       # Change to index when testing nationality
                    country2idx[country] = int(index)
                    idx2country[int(index)] = country
            elif file_name == 'country_to_ethnicity.txt':
                for k, line in enumerate(data):
                    file_len = k + 1
                    country, eth1, eth2 = line[:-1].split('\t')
                    country2ethnicity[int(country)] = [int(eth1), int(eth2)]
            elif 'data_' in file_name:
                for k, line in enumerate(data):
                    name, nationality = line[:-1].split('\t')
                    name = re.sub(r'\ufeff', '', name)    # delete BOM

                    unigram_vector = [unigram2idx[c] if c in unigram2idx else 0 for c in name]
                    bigram_vector= [bigram2idx[c1 + c2] if (c1+c2) in bigram2idx else 0
                            for c1, c2 in zip(*[name[i:] for i in range(2)])]
                    trigram_vector= [trigram2idx[c1 + c2 + c3] if (c1+c2+c3) in trigram2idx else 0 
                            for c1, c2, c3 in zip(*[name[i:] for i in range(3)])]

                    # label vector
                    nationality = country2idx[nationality]
                    if is_ethnicity:
                        ethnicity = country2ethnicity[nationality][1]
                        if ethnicity < 0:
                            continue
                    name_length = len(name)

                    if name_max_len < len(name):
                        name_max_len = len(name)

                    unigram_set.append(unigram_vector)
                    bigram_set.append(bigram_vector)
                    trigram_set.append(trigram_vector)
                    length_set.append(name_length)
                    if is_ethnicity:
                        labels.append(ethnicity)
                    else:
                        labels.append(nationality)
                    file_len = k + 1

                if 'train_ch' in file_name:
                    train_set = [unigram_set, bigram_set, trigram_set, length_set, labels]
                elif 'val' in file_name:
                    valid_set = [unigram_set, bigram_set, trigram_set, length_set, labels]
                elif 'ijcai' in file_name: # test
                    test_set = [unigram_set, bigram_set, trigram_set, length_set, labels]
                else:
                    assert True, 'not allowed file name %s'% file_name
                
                unigram_set = []
                bigram_set = []
                trigram_set = []
                length_set = []
                labels = []
            else:
                print('ignoring file', file_name)

            print('reading', file_name, 'of length', file_len)

    print('total data length:', len(train_set[0]), len(valid_set[0]), len(test_set[0]))
    print('shape of data:', np.array(train_set).shape, np.array(valid_set).shape, np.array(test_set).shape)
    print('name max length:', name_max_len)

    return (train_set, valid_set, test_set,
            [idx2unigram, unigram2idx, idx2country, country2ethnicity, idx2bigram, idx2trigram])


def get_char2vec(train_set, dim_embed, idx2char):
    sentences = []
    for sentence in train_set:
        char_seq = [idx2char[c] for c in sentence]
        sentences.append(char_seq)

    model = gensim.models.Word2Vec(sentences, size=dim_embed, window=5, min_count=0, iter=10)
    initializer = np.zeros((len(idx2char), dim_embed), dtype=np.float32)

    for idx in range(len(idx2char)):
        if idx2char[idx] in model:
            initializer[idx] = model[idx2char[idx]]
   
    '''
    for alphabet in idx2char.values():
        print('most similar to', alphabet, end=' is ')
        try:
            print(' '.join([(s) for s, _ in model.most_similar(positive=[alphabet], topn=5)]))
        except:
            print('no values', alphabet)
    '''
    
    return initializer


def get_data(params):
    ethnicity_dir = params['data_dir']
    is_valid = params['is_valid']
    train_set, valid_set, test_set, dictionary = get_ethnicity_data(ethnicity_dir, params)

    print(train_set[0][0])
    print(train_set[1][0])
    print(train_set[2][0])
    print(train_set[3][0], train_set[4][0])

    if not is_valid:
        train_set[0] = np.append(train_set[0], valid_set[0], axis=0)
        train_set[1] = np.append(train_set[1], valid_set[1], axis=0)
        train_set[2] = np.append(train_set[2], valid_set[2], axis=0)
        train_set[3] = np.append(train_set[3], valid_set[3], axis=0)
        train_set[4] = np.append(train_set[4], valid_set[4], axis=0)
    print('shape of data:', np.array(train_set).shape, np.array(valid_set).shape, np.array(test_set).shape)
    print('preprocessing done\n')
    
    return train_set, valid_set, test_set, dictionary


def experiment(model, dataset, params):
    print('## Training')
    valid_epoch = 1
    test_epoch = 1
    max_top1 = 0
    min_loss = 99999
    max_top5 = 0
    max_top1_epoch = 0
    nochange_cnt = 0
    early_stop = 5
    checkpoint_dir = params['checkpoint_dir']
    continue_train = params['continue_train']
    train_epoch = params['train_epoch']
    is_save = params['save']
    is_valid = params['is_valid']
    sess = model.session

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    if continue_train is not False:
        model.load(checkpoint_dir)

    start_time = time.time()
    for epoch_idx in range(train_epoch):
        train_cost, train_acc, train_acc5 = run(model, params, dataset[0], is_train=True)
        print("\nTraining loss: %.3f, acc1: %.3f, acc5: %.3f, ep: %d" % (train_cost, train_acc,
            train_acc5, epoch_idx))

        if (epoch_idx % valid_epoch == 0 or epoch_idx == train_epoch - 1) and is_valid:
            valid_cost, valid_acc, valid_acc5 = run(model, params, dataset[1], is_valid=is_valid)
            print("\nValidation loss: %.3f, acc1: %.3f, acc5: %.3f, ep: %d" % (valid_cost, valid_acc,
                valid_acc5, epoch_idx))
            if valid_acc > max_top1:
                max_top1 = valid_acc
                max_top5 = valid_acc5
                max_top1_epoch = epoch_idx
                nochange_cnt = 0
            else:
                nochange_cnt += 1
        elif not is_valid:
            if train_cost < min_loss:
                min_loss = train_cost
                nochange_cnt = 0
            else:
                nochange_cnt += 1

        if epoch_idx % test_epoch == 0 or epoch_idx == train_epoch - 1:
            test_cost, test_acc, test_acc5 = run(model, params, dataset[2], dataset[3], is_test=True)
            print("Testing loss: %.3f, acc1: %.3f, acc5: %.3f" % (test_cost, test_acc,
                test_acc5))
            print()
            if is_save:
                model.save(checkpoint_dir, sess.run(model.global_step))

        if nochange_cnt == early_stop:
            print("Early stopping applied\n")
            test_cost, test_acc, test_acc5 = run(model, params, dataset[2], dataset[3], is_test=True)
            print("Testing loss: %.3f, acc1: %.3f, acc5: %.3f" % (test_cost, test_acc,
                test_acc5))
            break

        # summary = sess.run(model.merged_summary, feed_dict=feed_dict)
        # model.train_writer.add_summary(summary, step)

    # model.save(checkpoint_dir, sess.run(model.global_step))
    model.reset_graph()
    return max_top1, max_top5, max_top1_epoch


def run(model, params, dataset, dictionary=None, is_train=False, is_valid=False, is_test=False):
    batch_size = params['batch_size']
    lstm_dropout = params['lstm_dropout']
    hidden_dropout = params['hidden_dropout']
    output_size = params['dim_output']
    max_time_step = params['max_time_step']
    sess = model.session
    cnt = 0.0
    total_cost = 0.0
    total_acc = 0.0
    total_acc5 = 0.0
    total_pred = None
    
    unigram_set, bigram_set, trigram_set, lengths, labels = dataset
    if is_valid or is_test:
        lstm_dropout = 1.0
        hidden_dropout = 1.0

    for datum_idx in range(0, len(unigram_set), batch_size):
        batch_unigram = unigram_set[datum_idx:datum_idx+batch_size]
        batch_bigram = bigram_set[datum_idx:datum_idx+batch_size]
        batch_trigram = trigram_set[datum_idx:datum_idx+batch_size]
        batch_lengths= lengths[datum_idx:datum_idx + batch_size]
        batch_labels = labels[datum_idx:datum_idx+batch_size]

        batch_unigram_onehot = []
        batch_bigram_onehot = []
        batch_trigram_onehot = []
        for unigram in batch_unigram:
            unigram_onehot = unigram
            while len(unigram_onehot) != max_time_step:
                unigram_onehot.append(0)
            batch_unigram_onehot.append(unigram_onehot)
        for bigram in batch_bigram:
            bigram_onehot = bigram
            while len(bigram_onehot) != max_time_step:
                bigram_onehot.append(0)
            batch_bigram_onehot.append(bigram_onehot)
        for trigram in batch_trigram:
            trigram_onehot = trigram
            while len(trigram_onehot) != max_time_step:
                trigram_onehot.append(0)
            batch_trigram_onehot.append(trigram_onehot)

        feed_dict = {model.unigram: batch_unigram_onehot, model.bigram: batch_bigram_onehot,
                model.trigram: batch_trigram_onehot, 
                model.lengths: batch_lengths, model.labels: batch_labels, 
                model.lstm_dropout: lstm_dropout, model.hidden_dropout: hidden_dropout}
        pred, cost, step = sess.run([model.logits, model.losses, model.global_step], feed_dict=feed_dict)

        if is_train:
            sess.run(model.optimize, feed_dict=feed_dict)
        
        if (datum_idx % (batch_size*5) == 0) or (datum_idx + batch_size >= len(unigram_set)):
            acc = accuracy_score(batch_labels, pred)
            acc5 = top_n_acc(batch_labels, pred, 5)
            _progress = progress((datum_idx + batch_size) / float(len(unigram_set)))
            _progress += " tr loss: %.3f, acc1: %.3f, acc5: %.3f" % (cost,
                    acc, acc5)
            if is_train:
                sys.stdout.write(_progress)
                sys.stdout.flush()
            cnt += 1
            total_cost += cost
            total_acc += acc
            total_acc5 += acc5
            
        if total_pred is None:
            total_pred = pred
        else:
            total_pred = np.append(total_pred, pred, axis=0)
    
    is_ethnicity = params['ethnicity']
    if is_test and not is_ethnicity:
        save_result(total_pred, lengths, labels, unigram_set, dictionary, params['pred_result_path'])
    if is_test and is_ethnicity:
        save_detail_result(total_pred, labels, lengths, unigram_set, dictionary, params['detail_result_path'])

    return total_cost / cnt, total_acc / cnt, total_acc5 / cnt


def accuracy_score(labels, logits):
    correct_prediction = np.equal(labels, np.argmax(logits, 1))
    accuracy = np.mean(correct_prediction.astype(float))
    return accuracy


def top_n_acc(labels, logits, top):
    top_n_logits = [logit.argsort()[-top:][::-1] for logit in logits]
    correct_prediction = np.array([(pred in topn) for pred, topn in zip(labels, top_n_logits)])
    accuracy = np.mean(correct_prediction.astype(float))
    return accuracy


def save_result(logits, indexes, labels, inputs, dictionary, path):
    idx2unigram, unigram2idx, idx2country, country2idx, _, _ = dictionary 
    top_n_logits = [logit.argsort()[-5:][::-1] for logit in logits]

    f = open(path, 'w')
    for logit, logit_index, label, input in zip(top_n_logits, indexes, labels, inputs):
        name = ''.join([idx2unigram[char] for char in input][:logit_index])
        pred = 'pred => ' + str(logit[0]) + ':' + idx2country[logit[0]] + '\n'
        pred += 'pred => ' + str(logit[1]) + ':' + idx2country[logit[1]] + '\n'
        pred += 'pred => ' + str(logit[2]) + ':' + idx2country[logit[2]] + '\n'
        pred += 'pred => ' + str(logit[3]) + ':' + idx2country[logit[3]] + '\n'
        pred += 'pred => ' + str(logit[4]) + ':' + idx2country[logit[4]] + '\n'
        corr = 'real => ' + str(label) + ':' + idx2country[label]
        result = '[correct]' if logit[0] == label else '[wrong]'
        end = '--------------------------------------------'
        f.write(result + '\n' + name + '\n' + pred + '\n' + corr + '\n' + end + '\n')
    f.close()


def save_detail_result(logits, labels, indexes, inputs, dictionary, path):
    idx2unigram, _, idx2country, country2ethnicity, _, _ = dictionary
    tp = dict()
    fp = dict()
    fn = dict()
    tn = dict()

    f = open(path, 'w')
    for ethnicity in range(13):
        key = ethnicity
        tp[key] = 0.0
        fp[key] = 0.0
        fn[key] = 0.0
        tn[key] = 0.0
        for logit, label in zip(logits, labels):
            if np.argmax(logit, 0) == key:
                if label == key:
                    tp[key] += 1
                else:
                    fp[key] += 1
            else:
                if label == key:
                    fn[key] += 1
                else:
                    tn[key] += 1
        if tp[key] == 0:
            continue
        pr = tp[key] / (tp[key] + fp[key])
        rc = tp[key] / (tp[key] + fn[key])
        f1 = 2*pr*rc / (pr+rc)

        f.write(str(ethnicity) + '\t%.2f\t%.2f\t%.2f'% (pr, rc, f1) + '\n')
    f.write('acc %.2f\n'% ((np.sum(list(tp.values())) + np.sum(list(tn.values()))) \
            / (np.sum(list(tp.values())) + np.sum(list(fp.values())) + np.sum(list(fn.values())) +
                np.sum(list(tn.values())))))
    f.close()
                    
