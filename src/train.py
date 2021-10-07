from initialisation import *
from saved_models import get_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# TODO: Clean up train.py


def train(p):
    """Train all the saved_models we want to compare"""
    """global count, saved
    if count < saved - 5:
        count += 5
        return"""

    # Filename is where the trained saved_models are saved
    filename = p["filename"]

    for i in range(KFOLDS):
        """if count < saved:
            count += 1
            continue"""

        print(filename, i)

        answers, embeddings = load_arrays()

        # Get the training and Validation Data
        sequences = filename[7:9] + "_sequences"
        scores = filename[7:9] + "_scores"
        x_train, x_valid, y_train, y_valid = get_train_sequences(i, answers[sequences], answers[scores])

        # Load the correct embeddings
        emb = filename[7:9] + "_" + p["6_emb"]

        t = False  # Whether to train embeddings or not
        if p["2_train"] == "train":
            t = True

        model = get_model(t, p["3_rnn"], p["4_bi"], embeddings[emb], p["5_att"], p["1_question"])

        # Simple early stopping
        es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=5)

        # Save the best model via checkpoints
        mc = ModelCheckpoint(filename + str(i) + ".h5",
                             monitor='val_accuracy',
                             mode='max',
                             verbose=1,
                             save_best_only=True)

        model.fit(x_train, y_train, epochs=30, batch_size=1,
                  validation_data=(x_valid, y_valid),
                  verbose=1,
                  callbacks=[es, mc])

        # count += 1


if __name__ == '__main__':
    count = 0
    # saved = num_models() - 1
    for params in CURR:
        train(params)
