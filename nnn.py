# coding: utf-8

import numpy as np
import copy
import pickle
from tqdm import tqdm


class NonmonotoneNeuralNetwork(object):
    """http://volga.esys.tsukuba.ac.jp/~mor/paper/mor1995.pdf"""
    def __init__(self, size=400, time_constant=5.0, initial_beta=0.4):
        self.size = size
        self.weight = np.zeros([size, size])

        self.tau_activation = time_constant
        self.tau_weight = 5000 * self.tau_activation
        self.initial_beta = initial_beta


    def save(self, path="save.pkl"):
        params = {
            "size": self.size,
            "weight": self.weight
        }
        pickle.dump(params, open(path, "wb"))


    def load(self, path="save.pkl"):
        params = pickle.load(open(path, "rb"))
        self.size = params["size"]
        self.weight = params["weight"]


    def partial_fit(self, x, loop=20, alpha=2.0):
        beta = np.copy(self.initial_beta)

        for it in tqdm(range(loop), desc="fit"):
            #print("iter:%d/%d beta:%f" % (it, loop, beta))

            for pi in range(x.shape[0]):
                stimulus = x[pi, :]
                if pi == 0:
                    activation = stimulus * self.initial_beta # x[0, :] * self.initial_beta
                else:
                    activation = self.__update_activation(activation, nonmonotone_output, beta, stimulus)
                binarized_output, nonmonotone_output = self.__update_output(activation)
                self.__update_weight(activation, nonmonotone_output, binarized_output, alpha, stimulus)

            beta -= self.initial_beta / loop


    def predict(self, stimulus, loop=0, stop_threshold=0.001):
        assert isinstance(stimulus, list)
        stimulus = np.array(stimulus).astype(np.float)
        activation = stimulus * self.initial_beta
        binarized_output, nonmonotone_output = self.__update_output(activation)
        last_activation = activation
        predictions = [copy.deepcopy(activation)]
        if loop == 0:
            loop = 100

        for it in range(loop):
            #print("iter:%d/%d" % (it, loop))
            activation = self.__update_activation(activation, nonmonotone_output)
            binarized_output, nonmonotone_output = self.__update_output(activation)
            predictions.append(copy.deepcopy(activation))

            if loop == 100 and np.sum(np.abs(activation - last_activation)) / float(self.size) < stop_threshold:
                break
            last_activation = activation

        return predictions


    # private


    def __update_output(self, activation):
        return self.__sign(activation), self.__output(activation)


    def __update_weight(self, activation, nonmonotone_output, binarized_output, alpha, stimulus):
        delta_error = stimulus - activation

        self.weight = ((self.tau_weight-1) * self.weight
                        + alpha * np.tile(delta_error, (self.size, 1)).T * nonmonotone_output
                       ) / self.tau_weight

        mask = np.ones((self.size, self.size)) - np.eye(self.size)
        self.weight *= mask


    def __update_activation(self, activation, output, beta=0, stimulus=None):
        weighted_input = np.dot(self.weight, output)
        if beta > 0.0:  # train
            activation = ((self.tau_activation - 1) * activation + weighted_input + beta * stimulus) / self.tau_activation
        else:
            activation = ((self.tau_activation - 1) * activation + weighted_input) / self.tau_activation
        return activation


    __C = -50.0
    __C_DASH = 10.0
    __H = 0.5
    __KAI = 1.0

    def __output(self, x):
        stimulus_abs = np.abs(x)
        e_c_i = np.exp(self.__C * x)
        e_cd_i = np.exp(self.__C_DASH * (stimulus_abs - self.__H))
        return ((1.0 - e_c_i) / (1.0 + e_c_i)) * ((1.0 - self.__KAI * e_cd_i) / (1.0 + e_cd_i))

    def __sign(self, x):
        y = np.ones(len(x))
        y[x < 0] = -1.0
        return y

