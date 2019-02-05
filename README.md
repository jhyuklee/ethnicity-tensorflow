Name Nationality Classification with Recurrent Neural Networks
==============================================================
Implementation of the paper, [Name Nationaltiy Classification with Recurrent Neural Networks](https://www.ijcai.org/proceedings/2017/0289) (Lee et al., IJCAI 2017)

## Requirements
* Python 3.4.3
* Tensorflow 1.0.1 (GPU enabled)
* Gensim

## Directories
* data/ : Olympic Name Datasets including raw and cleaned version
* main.py : Running RNN model with adjustable hyperparameters
* model.py : Model structure for RNN-LSTM
* ops.py : Tensorflow ops used in model.py
* dataset.py : Dataset reading and experiment workflow
* char2vec.py : Making pretrained char2vec
* preprocess.py : Preprocessing crawled dataset
* utils.py : Utility for printing

## Run the code
```bash
$ python main.py
```
