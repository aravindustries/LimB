#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm

# Load data (first 16 columns = TX phase shifts)
def load_beam_codebook(file_path):
    df = pd.read_csv(file_path, header=None)
    return df.iloc[:, :16].values.astype(str)  # Use first 16 columns

# Convert hex to phase (0x00 to 0x3F → 0 to 2π radians)
def hex_to_phase(hex_str):
    return (int(hex_str, 16) / 0x3F) * 1.9 * np.pi

# Simulate beam pattern for a 2x8 planar array
def simulate_beam_pattern_2x8(weights, freq=60e9):
    wavelength = 3e8 / freq
    spacing = wavelength / 2
    theta = np.linspace(-np.pi/2, np.pi/2, 180)  # Azimuth
    phi = np.linspace(0, np.pi, 90)              # Elevation
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    
    # 2x8 array positions
    x = np.arange(8) * spacing  # 8 columns
    y = np.arange(2) * spacing  # 2 rows
    xx, yy = np.meshgrid(x, y)
    positions = np.vstack([xx.ravel(), yy.ravel()]).T
    
    # Reshape weights to 2x8 and convert to complex phase shifts
    assert len(weights) == 16, f"Expected 16 phase weights, got {len(weights)}"
    phase_weights = np.reshape([hex_to_phase(w) for w in weights], (2, 8))
    complex_weights = np.exp(1j * phase_weights)  # e^(j*phase)
    
    # Array response
    beam_pattern = np.zeros_like(theta_grid)
    for i, t in enumerate(theta):
        for j, p in enumerate(phi):
            wave_vector = 2 * np.pi * np.array([np.sin(t) * np.cos(p), 
                                              np.sin(t) * np.sin(p)]) / wavelength
            phase_delays = np.exp(1j * (positions @ wave_vector))
            weighted_response = phase_delays * complex_weights.ravel()
            beam_pattern[j, i] = np.abs(np.sum(weighted_response))
    
    return np.degrees(theta_grid), np.degrees(phi_grid), beam_pattern

# Plot beam pattern
def plot_beam_2d(theta, phi, pattern, title):
    plt.figure(figsize=(10, 6))
    plt.contourf(theta, phi, 20 * np.log10(pattern + 1e-10), levels=50, cmap=cm.plasma)
    plt.colorbar(label="Gain (dB)")
    plt.title(title)
    plt.xlabel("Azimuth (degrees)")
    plt.ylabel("Elevation (degrees)")
    plt.show()

# Load and plot
beam_weights_ideal = load_beam_codebook("Antenna_beambook_ideal.csv")
beam_weights_calib = load_beam_codebook("Antenna_beambook.csv")

beam_idx = 32  # Beam index (0 to 63)
theta, phi, pattern_ideal = simulate_beam_pattern_2x8(beam_weights_ideal[beam_idx])
plot_beam_2d(theta, phi, pattern_ideal, f"Ideal Beam {beam_idx} (2x8 Array)")

theta, phi, pattern_calib = simulate_beam_pattern_2x8(beam_weights_calib[beam_idx])
plot_beam_2d(theta, phi, pattern_calib, f"Calibrated Beam {beam_idx} (2x8 Array)")
# %%
