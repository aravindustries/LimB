import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load data
csv_file = "../data/gain_profiles/gain_profiles_march23.csv" 
df = pd.read_csv(csv_file, header=None)

# Extract columns
angs = df[0].to_numpy()
tx_powers = df[1].to_numpy()
beams = np.arange(0, 63)
gains = df.iloc[:, 2:65].to_numpy()

# Filter out specific power levels
df_filtered = df[(df[1] != 10) & (df[1] != 56)]
df_filtered.to_csv("filtered_data3.csv", index=False, header=False)

# Get unique power levels after filtering
power_levels = np.unique(df_filtered[1].to_numpy())

# Save each power level's filtered data separately
for power in power_levels:
    df_power_filtered = df_filtered[df_filtered[1] == power]
    df_power_filtered.to_csv(f"plot3d/filtered_data_power_{power}.csv", index=False)

    # Extract corresponding angle and gain data
    mask = tx_powers == power
    angs_filtered = angs[mask]
    gains_filtered = gains[mask, :]  # Keep the unnormalized gains

    # Meshgrid for plotting
    X, Y = np.meshgrid(beams, angs_filtered)

    # Plot 3D surface
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_surface(X, Y, gains_filtered, cmap='jet', edgecolor='none')  # No normalization
    
    ax.set_xlabel("Beam Index")
    ax.set_ylabel("Angle")
    ax.set_zlabel("Gain")
    ax.set_title(f"3D Beam Profile Plot (Power {power})")
    plt.savefig(f'plot3d/plot3d{power}.png')
    plt.close(fig)  # Close the figure to free memory