#-*- coding: utf-8 -*-:
from __future__ import print_function

import codecs
import re
import numpy as np

from random import shuffle

category = 'raw'
clean = False
write = True
dataset_path = './data/crawl/countryResult.txt'
unigram_set_path = './data/' + category + '/0_unigram_to_idx.txt'
bigram_set_path = './data/' + category + '/1_bigram_to_idx.txt'
trigram_set_path = './data/' + category + '/2_trigram_to_idx.txt'
country_set_path = './data/' + category + '/country_to_idx.txt'
new_dataset_path = './data/' + category + '/data_'

dataset = codecs.open(dataset_path, 'r', 'utf-8').readlines()
name_to_country = dict()
name_to_year = dict()
unigram_set = set()
bigram_set = set()
trigram_set = set()
country_cnt = dict()

allowed_char = ['.', '\'']
cleaning_char = [':', '©', '¶']

for k, line in enumerate(dataset):
    datum = line.split('\t')
    country = datum[0]
    olympic_year = datum[1]
    medal = datum[2]
    record = datum[3]
    sports = datum[4]
    names_raw = datum[5][:-1]

    # more than 1 name in the list
    if len(names_raw.split('/')) >= 2:
        names_list = names_raw.split('/')[:-1]
        for name in names_list:

            if clean:
                # check for special characters 
                name = re.sub(r'[-–]', ' ', name)
                name = re.sub(r'\([^)]*\)', '', name)    
                name = re.sub(r'\"[^)]*\"', '', name)
                name = re.sub(r'\s\s', ' ', name)
                name = re.sub(r'[\s\t]$', '', name)
                name = name.lower()
                name = name.split(' ')
                name = ['$' + n + '$' if i < len(name) - 1 else '+' + n + '+' 
                        for i, n in enumerate(name)]
                name = ' '.join(name)
                 
                assert not name.endswith(' ') and not name.startswith(' ')
                assert not '  ' in name, (name, 'has double space!')

                if any(c in cleaning_char for c in name):
                    continue
            
            # if it's in the dict, check for year
            if name in name_to_country:    
                if country != name_to_country[name]:
                    saved_year = int(name_to_year[name].split(' ')[-1])
                    current_year = int(olympic_year.split(' ')[-1])
                    
                    # skip if older year
                    if  saved_year <= current_year:
                        # print('collision but not update (dict vs current)',
                        #         int(name_to_year[name].split(' ')[-1]), int(olympic_year.split(' ')[-1]))
                        continue
                    else: 
                        # print('collision and update (dict vs current)', 
                        #         int(name_to_year[name].split(' ')[-1]), int(olympic_year.split(' ')[-1]))
                        pass

            name_to_country[name] = country
            if country not in country_cnt:
                country_cnt[country] = 0
            country_cnt[country] += 1
            name_to_year[name] = olympic_year




if write:
    # write new dataset
    data_to_write = dict()
    train_data = dict()
    valid_data = dict()
    test_data = dict()
    name_to_country = list(name_to_country.items())
    shuffle(name_to_country)
    for name, country in name_to_country:
        if country_cnt[country] < 5 and clean:
            # print('small country', country, country_cnt[country])
            continue
        for char_idx, char in enumerate(name):
            unigram_set.add(char)
            if char_idx > 0:
                bigram_set.add(name[char_idx-1] + name[char_idx])
            if char_idx > 1:
                trigram_set.add(name[char_idx-2] + name[char_idx-1] + name[char_idx])
        if country not in train_data.values():
            train_data[name] = country
        else:
            data_to_write[name] = country

    data_size = len(train_data) + len(data_to_write)
    
    data_to_write = list(data_to_write.items())
    shuffle(data_to_write)
    for name, country in data_to_write:
        if len(train_data) < data_size * 0.6:
            train_data[name] = country
        elif len(valid_data) < data_size * 0.2:
            valid_data[name] = country
        else:
            test_data[name] = country
    
    new_dataset = open(new_dataset_path + category + '_train', 'w')
    for name, country in train_data.items():
        new_dataset.write(name + '\t' + country + '\n')
    new_dataset.close()
    new_dataset = open(new_dataset_path + category + '_valid', 'w')
    for name, country in valid_data.items():
        new_dataset.write(name + '\t' + country + '\n')
    new_dataset.close()
    new_dataset = open(new_dataset_path + category + '_test', 'w')
    for name, country in test_data.items():
        new_dataset.write(name + '\t' + country + '\n')
    new_dataset.close()

    # write unigram set
    unigram_dataset = open(unigram_set_path, 'wb')
    for idx, char in enumerate(sorted(unigram_set)):
        line = char + '\t' + str(idx) + '\n'
        unigram_dataset.write(line.encode('utf-8'))
    unigram_dataset.close()

    # write bigram set
    bigram_dataset = open(bigram_set_path, 'wb')
    for idx, char in enumerate(sorted(bigram_set)):
        line = char + '\t' + str(idx) + '\n'
        bigram_dataset.write(line.encode('utf-8'))
    bigram_dataset.close()

    # write trigram set
    trigram_dataset = open(trigram_set_path, 'wb')
    for idx, char in enumerate(sorted(trigram_set)):
        line = char + '\t' + str(idx) + '\n'
        trigram_dataset.write(line.encode('utf-8'))
    trigram_dataset.close()

    # write country set
    country_size = len(country_cnt)
    country_dataset = open(country_set_path, 'wb')
    write_idx = 0
    for idx, (country, cnt) in enumerate(sorted(country_cnt.items())):
        line = country + '\t' + str(idx) + '\n'
        if cnt < 5 and clean:
            country_size -= 1
            continue
        country_dataset.write(line.encode('utf-8'))
        write_idx += 1
    country_dataset.close()

print('\ndataset size', data_size)
print('train test valid size', len(train_data), len(valid_data), len(test_data))
print('sample data', name_to_country[532], name_to_country[15])
print('unigram set', len(unigram_set))
print('bigram set', len(bigram_set))
print('trigram set', len(trigram_set))
print('country set', country_size)

