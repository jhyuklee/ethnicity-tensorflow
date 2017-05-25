Name Nationality Classification with Recurrent Neural Networks
==============================================================
Implementation of Name Nationality Classification on Olympic Name Dataset

Paper will be published at IJCAI 2017 titled as:
[Name Nationaltiy Classification with Recurrent Neural Networks]

- Jinhyuk Lee (jinhyuk\_lee@korea.ac.kr)
- Hyunjae Kim (gamica@sogang.ac.kr)
- Miyoung Ko (gomi1503@korea.ac.kr)
- Donghee Choi (choidonghee@korea.ac.kr)
- Jaehoon Choi (jchoi@kono.ai)
- Jaewoo Kang (kangj@korea.ac.kr)

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
