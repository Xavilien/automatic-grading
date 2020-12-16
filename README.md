# Automatic Grading of Online Formative Assessments using Bidirectional Neural Networks and Attention Mechanism
### Description
Source code for a project on the automatic grading of online formative assessments

### Overview of Source Code files
**processing.py**: clean and process the training dataset as well as the word embeddings. Saves arrays so that they do not have to be regenerated each time.

**initialisation.py**: generate the hyperparameters for all the models we want to train

**models.py**: return a Keras model given the hyperparameters

**train.py**: train all models

**evaluate.py**: generate the results from the trained models

**split_test.py**: compare the performance of the best models against number of training samples 

**attention.py**: plot attention weights for a particular model given a sample answer

### How to run code for quantitative component
1. Run processing.py to process the data (should be run only once)
2. Run train.py to train all the different types of models (5 times each)
3. Run evaluate.py to evaluate the models trained in 2.
4. Run split_test.py to train the two best models
5. Run attention.py to visualise the attention weights for a specific answer