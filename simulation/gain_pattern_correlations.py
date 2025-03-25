import matplotlib.pyplot as plt
import numpy as np


d = 0.55
Nr = 16

lim = 32
k = 8


def get_corr(betas):
    alphas = np.array(range(-lim, lim+1)) * np.pi / 180
    alpha_matrix = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(alphas).reshape(-1, 1))
    beta_matrix = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(betas * np.pi / 180).reshape(-1, 1))
    gain = alpha_matrix @ beta_matrix.conj().T / Nr
    gain = np.real(gain)
    return gain @ gain.T

def plot(corr, bool=False):
    if bool:
        rows, cols = corr.shape
        x = np.arange(cols)
        y = np.arange(rows)
        x, y = np.meshgrid(x, y)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, corr)

        ticklabels = np.array(range(-lim//k, (lim//k) + 1)) * k
        ticks = ticklabels + lim

        print(ticks)
        print(ticklabels)

        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        ax.set_xticklabels(ticklabels)
        ax.set_yticklabels(ticklabels)

        plt.show()
    else:
        plt.imshow(np.log10(corr), cmap='viridis', origin='lower')  # 'viridis' is a common colormap
        plt.colorbar()

        ticklabels = np.array(range(-lim//k, (lim//k) + 1)) * k
        ticks = ticklabels + lim

        print(ticks)
        print(ticklabels)

        plt.xticks(ticks, ticklabels)
        plt.yticks(ticks, ticklabels)

        plt.show()

def get_betas(spread, B):
    return np.linspace(-spread, spread, B)
    

plot(get_corr(get_betas(60, 16)))
