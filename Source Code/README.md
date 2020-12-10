# Automatic Grading of Online Formative Assessments using Bidirectional Neural Networks and Attention Mechanism

### Overview of Source Code files
_processing.py_: clean and process the training dataset as well as the word embeddings 

_initialisation.py_: generate the hyperparameters for all the models we want to train

_models.py_: return a Keras model given the hyperparameters

_train.py_: train all models

_evaluate.py_: generate the results from the trained models

_split_test.py_: compare the performance of the best models against number of training samples 

_attention.py_: plot attention weights for a particular model given a sample answer

_kfolds.py_: honestly not too sure what I was doing with this file but I think it's just an extra file that I wanted to try something else with but did not end up using
