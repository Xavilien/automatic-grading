from src.initialisation import *
from src import models

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Bidirectional, LSTM, Subtract
from tensorflow.keras.optimizers import Adam


def get_model(embeddings):
    input_layer = Input(shape=(None,))

    embedding_layer = Embedding(
        input_dim=len(embeddings),
        output_dim=300,
        weights=[embeddings],
        trainable=False,
        mask_zero=True)(input_layer)

    rnn_layer = LSTM(64, dropout=0.1)(embedding_layer)

    dense_layer = Dense(64, activation='relu')(rnn_layer)

    siamese = Model(inputs=input_layer, outputs=dense_layer)

    in_a = Input(shape=(None,))  # Anchor (marking scheme)
    in_b = Input(shape=(None,))  # Student answer

    emb_a = siamese(in_a)
    emb_b = siamese(in_b)

    subtract = Subtract()([emb_a, emb_b])

    dense_layer2 = Dense(64, activation='relu')(subtract)

    output = Dense(3, activation='softmax')(dense_layer2)

    model = Model(inputs=[in_a, in_b], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def main():
    answers, embeddings = load_arrays()
    x_train, x_valid, y_train, y_valid = get_train_sequences(1, answers['q1_sequences'], answers['q1_scores'])
    model = get_model(embeddings['q1_glove'])

    model.fit([x_train, x_train], y_train, epochs=10, batch_size=1)


if __name__ == '__main__':
    main()
