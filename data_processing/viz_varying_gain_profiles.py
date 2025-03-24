#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

csv_file = "../data/gain_profiles/vary_tx_pow_data.csv" 
df = pd.read_csv(csv_file, header=None)

angs = df[0].to_numpy()  # First column: angles
tx_powers = df[1].to_numpy()  # Second column: Tx power levels
beams = np.arange(0, 63)  # Beam indices (0 to 62)
gains = df.iloc[:, 2:65].to_numpy()  

# Normalize gain profiles

# Unique power levels
power_levels = np.unique(tx_powers)

for power in power_levels:
    mask = tx_powers == power
    angs_filtered = angs[mask]  # Angles for this power level
    gains_filtered = gains[mask, :]  # Corresponding normalized gains

    gains_norm = gains_filtered - np.min(gains_filtered)
    gains_norm = gains_norm / np.max(gains_norm)

    # meshgrid for 3D plotting
    X, Y = np.meshgrid(beams, angs_filtered)  # X: beam indices, Y: angles

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_surface(X, Y, gains_norm, cmap='jet', edgecolor='none')
    
    ax.set_xlabel("Beam Index")
    ax.set_ylabel("Angle")
    ax.set_zlabel("Normalized Gain")
    ax.set_title(f"3D Beam Profile Plot")
    plt.savefig(f'plot3d{power}.png') # save the plots
# %%
