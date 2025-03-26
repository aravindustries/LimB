#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def viz_gain_prof(csv_file):
    df = pd.read_csv(csv_file, header=None)

    angs = df[0].to_numpy()
    tx_powers = df[1].to_numpy()
    beams = np.arange(0, 63)
    gains = df.iloc[:, 2:65].to_numpy()

    norm_gains = (gains - np.min(gains)) / (np.max(gains) - np.min(gains))

    X,Y = np.meshgrid(beams, angs)

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, norm_gains, cmap='jet', edgecolor='none')

    ax.set_xlabel("beam")
    ax.set_ylabel("angle")
    ax.set_zlabel("Gain")
    ax.set_title("3D gain plot" + csv_file)

    plt.show()
    
viz_gain_prof("../data/gain_profiles/gain_profiles_march21.csv")

viz_gain_prof("../data/gain_profiles/gain_profiles_march22a.csv")

viz_gain_prof("../data/gain_profiles/gain_profiles_march22b.csv")

viz_gain_prof("../data/gain_profiles/gain_profiles_march23.csv")


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def viz_gain_prof(csv_file):
    df = pd.read_csv(csv_file, header=None)

    angs = df[0].to_numpy()
    tx_powers = df[1].to_numpy()
    beams = np.arange(0, 63)
    gains = df.iloc[:, 2:65].to_numpy()

    norm_gains = (gains - np.min(gains)) / (np.max(gains) - np.min(gains))

    X,Y = np.meshgrid(beams, angs)

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, norm_gains, cmap='jet', edgecolor='none')

    ax.set_xlabel("beam")
    ax.set_ylabel("angle")
    ax.set_zlabel("Gain")
    ax.set_title("3D gain plot" + csv_file)

    plt.show()
    
viz_gain_prof("../data_processing/multipath_mixed.csv")
# %%
