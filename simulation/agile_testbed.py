# This differs from agile.py in order to properly use testbed data

from generate_data import *

import matplotlib.pyplot as plt
import numpy as np


class Agile:
    def __init__(self, betas, Nr=8):
        self.d = 0.55
        self.Nr = Nr

        self.beta_matrix = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(betas * np.pi / 180).reshape(-1, 1))


    def call(self, X):
        k = 45

        # Get theoretical gains
        alphas = np.array(range(-k, k+1)) * np.pi / 180
        alpha_matrix = np.exp(-2j * np.pi * d * np.arange(self.Nr) * np.sin(alphas.reshape(-1, 1)))

        gain = alpha_matrix @ self.beta_matrix.conj().T / self.Nr
        gain = np.abs(np.square(gain))
        gain /= np.linalg.norm(gain, axis=1).reshape(-1, 1)  # Normalize

        idx = np.arange(X.shape[0])
        for i in range(X.shape[0]):
            # breakpoint()
            corr = X[i] @ gain.T
            idx[i] = np.argmax(corr)

        return idx - k


    def evaluate(self, X, y):
        y_hat = self.call(X)
        y_ref = y - 32
        # breakpoint()

        arr = np.abs(y_ref - y_hat)
        accuracy = (arr == 0).mean()
        mae = arr.mean()
        return accuracy, mae
