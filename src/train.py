from initialisation import GRID, KFOLDS, get_num_models, load_arrays, get_train_sequences
from models.models import get_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def train():
    """
    Train all the models we want to compare. For each of the hyperparameters we train 5 models via k-folds
    cross-validation.

    We count the number of models that have already been trained so that in the event that training is interrupted,
    the code can continue training from the last model it had been training, skipping the models that have already been
    trained.
    """
    count = 1
    num_models = get_num_models()  # number of models that have already been trained

    # We loop through each set of hyperparameters and train 5 models each via KFOLDS cross-validation
    for p in GRID:
        for i in range(KFOLDS):
            if count < num_models:  # so that we re-train the last model that was saved
                count += 1
                continue

            # Filename is where the trained models are saved in saved_models
            filename = p["filename"] / f"{i}.h5"
            print(f"Model {count}/{len(GRID) * 5}: {filename}")

            answers, embeddings = load_arrays()

            # Get the training and Validation Data
            sequences = f"q{p['1_question']}_sequences"
            scores = f"q{p['1_question']}_scores"
            x_train, x_valid, y_train, y_valid = get_train_sequences(i, answers[sequences], answers[scores])

            fit(embeddings, filename, p, x_train, x_valid, y_train, y_valid)

            count += 1


def fit(embeddings, filename, p, x_train, x_valid, y_train, y_valid):
    """
    Fit model to the data. Re-used in split_test.
    """
    model = get_model(p["2_train"] == "train",  # whether to train or freeze the embeddings
                      p["3_rnn"],  # whether to use an rnn layer or dense layer (for baseline models)
                      p["4_bi"],  # whether the rnn layer should be bidirectional
                      embeddings[f"q{p['1_question']}_{p['6_emb']}"],  # load correct embedding (glove/fasttext)
                      p["5_att"],  # whether to use an attention mechanism with rnn layer
                      p["1_question"])  # question number

    # Simple early stopping
    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=5)

    # Save the best model via checkpoints
    mc = ModelCheckpoint(filename,
                         monitor='val_accuracy',
                         mode='max',
                         verbose=1,
                         save_best_only=True)

    model.fit(x_train, y_train, epochs=30, batch_size=1,
              validation_data=(x_valid, y_valid),
              verbose=1,
              callbacks=[es, mc])


if __name__ == '__main__':
    train()
