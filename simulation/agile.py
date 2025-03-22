from generate_data import *

import matplotlib.pyplot as plt
import numpy as np


class Agile:
    def __init__(self, B, spread, Nr=16):
        self.d = 0.55
        self.Nr = Nr

        betas = np.linspace(-spread, spread, B)
        self.beta_matrix = np.exp(-2j * np.pi * d * np.arange(self.Nr) * np.sin(betas).reshape(-1, 1))


    # X, y = generate_data(100, snr_range=(10, 20), k_factor_range=(0,0))
    # breakpoint()
    def call(self, X):
        X = X[:, :self.Nr] + 1j * X[:, self.Nr:]
        beams =  self.beta_matrix.conj() @ X

        k = 32

        # Get theoretical gains
        alphas = np.array(range(-k, k+1)) * np.pi / 180
        alpha_matrix = np.exp(-2j * np.pi * d * np.arange(self.Nr) * np.sin(alphas.reshape(-1, 1)))

        gain = alpha_matrix @ self.beta_matrix.conj().T / self.Nr
        gain = np.abs(np.square(gain))  # the amount by which _power_ is amplified, not voltage
        # breakpoint()
        gain /= np.linalg.norm(gain, axis=1).reshape(-1, 1)  # Normalize

        # Get emperical gains
        # emp_gain = np.einsum('njk,nkj->nj', beams, beams.conj().transpose(0,2,1)) / np.square(beams.shape[2])
        # emp_gain = np.real(emp_gain)  # remove small complex parts resulting from quantization error
        emp_gain = np.var(beams, axis=2)
        # breakpoint()
        emp_gain /= np.linalg.norm(emp_gain, axis=1).reshape(-1, 1)  # Normalize

        # Get the correlation
        corr = np.einsum('jk,nk->nj', gain, emp_gain) / beams.shape[1]

        # And now estimate the angle
        idx = np.argmax(corr, axis=1)
        # breakpoint()
        return idx - k


    def evaluate(self, X, y):
        y_hat = self.call(X)
        y_ref = y - 32
        # breakpoint()

        arr = np.abs(y_ref - y_hat)
        accuracy = (arr == 0).mean()
        mae = arr.mean()
        return accuracy, mae
