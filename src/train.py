from initialisation import *
from models.models import get_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# TODO: Clean up train.py


def train(p):
    """Train all the saved_models we want to compare"""
    global count, num_models
    if count < num_models - 5:
        count += 5
        return

    # Filename is where the trained saved_models are saved
    filename = p["filename"]

    for i in range(KFOLDS):
        if count < num_models:
            count += 1
            continue

        print(filename, i)

        answers, embeddings = load_arrays()

        # Get the training and Validation Data
        sequences = f"q{p['1_question']}_sequences"
        scores = f"q{p['1_question']}_scores"
        x_train, x_valid, y_train, y_valid = get_train_sequences(i, answers[sequences], answers[scores])

        model = get_model(p["2_train"] == "train",  # whether to train or freeze the embeddings
                          p["3_rnn"],  # whether to use an rnn layer or dense layer (for baseline models)
                          p["4_bi"],  # whether the rnn layer should be bidirectional
                          embeddings[f"q{p['1_question']}_{p['6_emb']}"],  # Load correct embeddings (glove/fasttext)
                          p["5_att"],  # whether to use an attention mechanism with rnn layer
                          p["1_question"])  # question number

        # Simple early stopping
        es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=5)

        # Save the best model via checkpoints
        mc = ModelCheckpoint(filename/f"{i}.h5",
                             monitor='val_accuracy',
                             mode='max',
                             verbose=1,
                             save_best_only=True)

        model.fit(x_train, y_train, epochs=30, batch_size=1,
                  validation_data=(x_valid, y_valid),
                  verbose=1,
                  callbacks=[es, mc])

        count += 1


if __name__ == '__main__':
    # There is some logic here to count the number of models that have already been trained so that in the event that
    # training is interrupted, the code can continue training from the last model it had been training
    # we get the number of models already trained using get_num_models() (subtracting 1 so that we retrain the last
    # model since we cannot guarantee that training had been completed for that model). Then we use count to skip the
    # models that have already been trained until we get to the last model we had been training and continue from there
    count = 0
    num_models = get_num_models() - 1  # number of models that have already been trained

    # We loop through each set of hyperparameters and train 5 models each via KFOLDS cross-validation
    for params in CURR:
        train(params)
