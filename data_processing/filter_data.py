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
df_filtered = df_filtered[df_filtered[1] != 56]
df_filtered.to_csv("filtered_data3.csv", index=False, header=False)

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

# %%
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

    # Create DataFrame with beam gains
    power_data = pd.DataFrame(gains_filtered, columns=[f'Beam_{i}' for i in range(gains_filtered.shape[1])])
    
    # Insert 'Angle' and 'Power_Level' at the beginning (left side)
    power_data.insert(0, 'Power_Level', power)  # Now Power_Level is the first column
    power_data.insert(0, 'Angle', angs_filtered)
    power_data.to_csv(f'power_{power}_data.csv', index=False)

    X, Y = np.meshgrid(beams, angs_filtered)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_surface(X, Y, gains_norm, cmap='jet', edgecolor='none')
    
    ax.set_xlabel("Beam Index")
    ax.set_ylabel("Angle")
    ax.set_zlabel("Normalized Gain")
    ax.set_title(f"3D Beam Profile Plot (Power {power})")
    plt.savefig(f'plot3d_{power}.png')
    plt.show()
    plt.close()
# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

input_dir = "."
output_dir = "beam_plots"
target_angle = 12

os.makedirs(output_dir, exist_ok=True)
csv_files = glob.glob(f"{input_dir}/power_*_data.csv")
plt.figure(figsize=(12, 8))

for file in csv_files:
    df = pd.read_csv(file)
    power_level = file.split('_')[1].split('.')[0]
    angles = df['Angle'].unique()
    closest_angle = angles[np.argmin(np.abs(angles - target_angle))]
    angle_data = df[np.isclose(df['Angle'], closest_angle)]
    beam_columns = [col for col in df.columns if col.startswith('Beam_')]
    beam_gains = angle_data[beam_columns].values.flatten()
    
    beam_gains_normalized = (beam_gains - np.min(beam_gains)) / (np.max(beam_gains) - np.min(beam_gains))
    
    beam_numbers = np.arange(len(beam_gains_normalized))
    plt.plot(beam_numbers, beam_gains_normalized, label=f'SNR {20*np.log(int(power_level)/300)}')

plt.xlabel('Beam Index')
plt.ylabel('Normalized Gain')
plt.title(f'Normalized Beam Patterns at Angle = {closest_angle}°')
plt.grid(True)
plt.legend()
plt.tight_layout()

output_file = f"{output_dir}/normalized_beams_at_angle_{target_angle}.png"
plt.savefig(output_file, dpi=300)
plt.show()
# %%
