#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

csv_file = "../data/gain_profiles/vary_tx_pow_data.csv" 
df = pd.read_csv(csv_file, header=None)

angs = df[0].to_numpy()
tx_powers = df[1].to_numpy()
beams = np.arange(0, 63)
gains = df.iloc[:, 2:65].to_numpy()

df_filtered = df[df[1] != 10]
df_filtered.to_csv("filtered_data.csv", index=False, header=False)

power_levels = np.unique(df_filtered[1].to_numpy())

for power in power_levels:
    mask = tx_powers == power
    angs_filtered = angs[mask]
    gains_filtered = gains[mask, :]

    gains_norm = gains_filtered - np.min(gains_filtered)
    gains_norm = gains_norm / np.max(gains_norm)

    X, Y = np.meshgrid(beams, angs_filtered)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_surface(X, Y, gains_norm, cmap='jet', edgecolor='none')
    
    ax.set_xlabel("Beam Index")
    ax.set_ylabel("Angle")
    ax.set_zlabel("Normalized Gain")
    ax.set_title(f"3D Beam Profile Plot (Power {power})")
    plt.savefig(f'plot3d{power}.png')
