# coding: utf-8

import numpy as np
import gensim
import threading
from flask import Flask, jsonify, request
import json

PRETRAINED_W2V_PATH = './model.bin'

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

model = gensim.models.KeyedVectors.load_word2vec_format(PRETRAINED_W2V_PATH, binary=True)

@app.route('/word_vector', methods=['GET'])
def word_vector():
    word = request.args.get('word')
    vector = np.array(model[word]).astype(float).tolist()
    return jsonify({'vector': vector}), 200


if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0', port=8888)

