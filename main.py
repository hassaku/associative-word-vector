# coding: utf-8

import requests
import numpy as np
import pandas as pd
import json

from sklearn.random_projection import GaussianRandomProjection
from nnn import NonmonotoneNeuralNetwork


def trajectory_patterns(que, target, steps=30):
    assert isinstance(que, list)
    assert len(que) % steps == 0
    pattern = np.copy(que)
    target = np.copy(target)
    patterns = [np.copy(que)]

    batch_size = int(len(que) / steps)

    indices = np.random.permutation(len(que))
    for s in range(0, len(que)+1, batch_size):
        pattern[indices[s:s+batch_size]] = target[indices[s:s+batch_size]]
        patterns.append(np.copy(pattern))

    patterns.append(np.copy(target))
    return np.vstack(patterns)


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def similarities(prediction, patterns):
    sims = {}
    for word, vector in patterns.items():
        sims[word] = cos_sim(prediction, vector)
    return sims


def get_word_vector(word):
    try:
        res_json = requests.get(
            "http://0.0.0.0:8888/word_vector?word={word}".format(word=word)).json()
    except json.decoder.JSONDecodeError:
        return None

    vector = res_json["vector"]
    return vector


def get_vectors(words, dims):
    word_vectors = []
    for word in words:
        word_vectors.append(get_word_vector(word))

    # convert vectors with specific dimension
    g = GaussianRandomProjection(dims)
    g.fit(np.array(word_vectors))
    random_mat = g.components_.transpose()

    vectors = {}
    for word, word_vector in zip(words, word_vectors):
        vectors[word] = g.transform(np.array([word_vector]))[0].tolist()

    return vectors


def train(nnn, vectors, df):
    print(df)

    for _ in range(5):
        for _, row in df.iterrows():
            nnn.partial_fit(trajectory_patterns(vectors[row.que], vectors[row.target]), loop=5)

    nnn.save()


def test(nnn, vectors, df, loop=30):
    ques = []
    answers = []
    targets = []
    for _, row in df.iterrows():
        predictions = nnn.predict(vectors[row.que], loop=loop)

        inferred_words = []
        for i, prediction in enumerate(predictions):
            sims = similarities(prediction, vectors)
            rankings = sorted(sims.items(), key=lambda x:x[1], reverse=True)

            if (len(inferred_words) == 0) or (inferred_words[-1] != rankings[0][0]):
                inferred_words.append(rankings[0][0])

        print("que: {que} target: {target} through: {inferred}".format(
            que=row.que, target=row.target, inferred=inferred_words))

        ques.append(row.que)
        targets.append(row.target)
        answers.append(inferred_words[-1])

    df_result = pd.DataFrame()
    df_result["que"] = ques
    df_result["target"] = targets
    df_result["answer"] = answers
    print(df_result)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true')
    args = parser.parse_args()

    np.random.seed(123)

    df = pd.read_csv("data.csv")
    words = np.unique(df.que.unique().tolist() + df.target.unique().tolist())

    vectors = get_vectors(words, dims=30*30)
    nnn = NonmonotoneNeuralNetwork(size=len(list(vectors.values())[0]))

    if args.train:
        train(nnn, vectors, df[df.train == 1])
    else:
        nnn.load()
        print("closed test")
        test(nnn, vectors, df[df.train == 1])
        print("open test")
        test(nnn, vectors, df[df.train == 0])


