import tensorflow as tf
from tensorflow.keras.models import Model, load_model
import numpy as np
from models.models import Attention
import plotly.offline as py
import plotly.graph_objects as go
from pathlib import Path

FILEPATH = Path(__file__).parent.absolute()
scores = np.load(f"{FILEPATH}/arrays/q1/scores.npy")
sequences = np.load(f"{FILEPATH}/arrays/q1/sequences.npy")
answers = np.load(f"{FILEPATH}/arrays/q1/answers.npy", allow_pickle=True)


def get_attention(model, i):
    """Returns the attention weights given a model and sequence i"""
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.layers[2].output)
    x = intermediate_layer_model.predict(np.array([sequences[i]]))

    x = tf.convert_to_tensor(x, dtype=tf.float32)

    attention = model.layers[3]

    a = attention.get_attention_weights(x)

    return np.array(a[0])


def plot_attention(weights, answer, save=False):
    """Plots the attention weights for each word in an answer."""
    z = weights[-len(answer):]

    data = {}
    for i, word in enumerate(answer):
        if word in data.keys():
            data[word] += z[i]
        else:
            data[word] = z[i]

    data = list(zip(data.keys(), [data[i] for i in data.keys()]))
    data = sorted(data, key=lambda s: s[1])

    words = [i[0] for i in data]
    weights = [[i[1]] for i in data]

    plot1 = go.Heatmap(y=words, z=weights, colorscale="Blues")

    layout = go.Layout(
        title=dict(text="Attention Weights of Answer Sample from Dataset 1", x=0.5, xanchor="center"),
        xaxis=dict(title="Attention Weights", nticks=1, mirror=True, linecolor='black'),
        yaxis=dict(title="Words", ticks="outside", nticks=len(words), mirror=True, linecolor='black'),
        plot_bgcolor='black',
        margin=go.layout.Margin(l=0, r=0, b=0, t=0, ))

    fig = go.Figure(data=plot1, layout=layout)
    fig.update_xaxes(showticklabels=False)

    py.iplot(fig)

    if save:
        fig.write_image("attention.pdf", format="pdf")


if __name__ == '__main__':
    bilstm = load_model(f"{FILEPATH}/saved_models/attention_model.h5",
                        custom_objects={"Attention": Attention})

    index = 113
    plot_attention(get_attention(bilstm, index), answers[index])
