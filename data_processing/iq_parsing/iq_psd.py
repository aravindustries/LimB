#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.preprocessing import MinMaxScaler
import random

df = pd.read_csv('../../../beamtracking15/final_beam_iq.csv', header=None)

#print(df.head(14))

df.drop(df.columns[1], axis=1, inplace=True)

#print(df.iloc[45])

angles = df.iloc[:, 0].values  # First column (angle)
beams = df.iloc[:, 1].values   # Second column (beam)
complex_data = df.iloc[:, 2:].values  # Complex data values starting from the third column

#print(angles)
#print(beams)
#print(complex_data.shape)

# Get the unique angles and beams
unique_angles = np.unique(angles)
unique_beams = np.unique(beams)
num_values = complex_data.shape[1]

# Create a mapping for angles to indices
angle_map = {angle: idx for idx, angle in enumerate(unique_angles)}

# Initialize a 3D array with the shape (num_angles, num_beams, num_values)
result = np.zeros((len(unique_angles), len(unique_beams), num_values), dtype=complex)

# Fill the array with the complex numbers
for i, angle in enumerate(unique_angles):
    for j, beam in enumerate(unique_beams):
        # Mask to select rows where angle and beam match
        mask = (angles == angle) & (beams == beam)
        
        # Check if more than one row matches the (angle, beam) pair
        if np.sum(mask) == 1:
            result[i, j, :] = complex_data[mask, :].flatten()
        else:
            # If multiple rows match, you can average them or take the first one
            result[i, j, :] = complex_data[mask, :][0]  # Take the first row that matches

#print(result[0, 45, :])

def viz_iq_profile(result):
    gp = np.zeros((91, 63))
    for i in range(91):
        for j in range(63):
            gp[i, j] = np.var(result[i, j, :])

    angs = np.arange(0, 91)
    beams = np.arange(0, 63) 
    X, Y = np.meshgrid(beams, angs)

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, gp, cmap='jet', edgecolor='none')

    ax.set_xlabel("Beam")
    ax.set_ylabel("Angle")
    ax.set_zlabel("Gain")

#viz_iq_profile(result)

def single_mask(decay_rate=0.01, tolerance=2):
    mask = np.zeros((91, 63))

    A = 62
    B = -90
    C = 0
    norm_factor = np.sqrt(A**2 + B**2)  # Precompute normalization factor

    for i in range(91):
        for j in range(1, 63):
            distance = abs(A * i + B * j + C) / norm_factor
            
            # Ensure distance is explicitly a float
            distance = float(distance)

            # Debugging statement to check `distance` values
            # print(f"i={i}, j={j}, distance={distance}")

            # If the point is nearly on the line, set it exactly to 1
            if distance < tolerance:
                mask[i, j] = 1.0
                if i > 30 and i < 47:
                    if j > 20 and j < 37:
                        mask[i,j] = np.random.uniform(0.7, 1)
            else:
                mask[i, j] = np.exp(-decay_rate * distance) * random.random()
    mask[45, :] = np.zeros(63)
    idx = np.arange(91)
    distances = np.abs(idx - 45)
    mask[:, 0] = np.exp(-decay_rate * distances)
    # Ensure df has the correct number of rows before adding mask
    return mask


def apply_rayleigh_fading_and_noise(data, snr_db=10):

    num_angles, num_beams, num_values = data.shape
    
    # Normalize the complex data by its RMS value
    signal_power = np.mean(np.abs(data) ** 2)  # Average power before normalization
    data_norm = data / np.max(data)  # Normalize so power is 1

    # Generate Rayleigh fading coefficients (complex Gaussian)
    rayleigh_fading = (np.random.normal(0, 1, (num_angles, num_beams, num_values)) + 
                       1j * np.random.normal(0, 1, (num_angles, num_beams, num_values))) / np.sqrt(2)
    snr = 10 ** ((snr_db) / 10)

    # Compute noise power based on SNR
    noise_power = 1 / snr  # Since signal power is now 1

    # Generate complex Gaussian noise
    noise = np.sqrt(noise_power / 2) * (np.random.randn(num_angles, num_beams, num_values) + 
                                        1j * np.random.randn(num_angles, num_beams, num_values))

    # Apply Rayleigh fading and noise
    received_signal = rayleigh_fading * data_norm + noise

    return received_signal


def get_aug_data(snr):
    result_masked = np.zeros(result.shape)

    mask = single_mask()
    for i in range(1024):
        result_masked[:,:,i] = np.multiply(result[:,:,i], mask)

    snr_db = 10 # Define desired SNR
    result_noisy = apply_rayleigh_fading_and_noise(result_masked, snr_db)

    return result_noisy






