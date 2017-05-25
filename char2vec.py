import gensim

from dataset import get_ethnicity_data


data_dir = './data/raw'
params = {'ethnicity': False}
train_set, valid_set, test_set, dictionary = get_ethnicity_data(data_dir, params)
vec = 2
dic = 5

sentences = []
for sentence in train_set[vec][:]:
    char_seq = [dictionary[dic][c] for c in sentence]
    sentences.append(char_seq)
for sentence in valid_set[vec][:]:
    char_seq = [dictionary[dic][c] for c in sentence]
    sentences.append(char_seq)
for sentence in test_set[vec][:]:
    char_seq = [dictionary[dic][c] for c in sentence]
    sentences.append(char_seq)

model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=0, iter=100)

for alphabet in dictionary[dic].values():
    print('most similar to', alphabet, end=' is ')
    try:
        print(' '.join([(s) for s, _ in model.most_similar(positive=[alphabet], topn=5)]))
    except:
        print('no values', alphabet)
