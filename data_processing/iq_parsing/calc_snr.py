#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

# Load and process the data as before
df = pd.read_csv('../../../beamtracking17/combined_beam_iq.csv', header=None)
df.drop(df.columns[1], axis=1, inplace=True)

angles = df.iloc[:, 0].values
beams = df.iloc[:, 1].values
complex_data = df.iloc[:, 2:].values

unique_angles = np.unique(angles)
unique_beams = np.unique(beams)
num_values = complex_data.shape[1]

angle_map = {angle: idx for idx, angle in enumerate(unique_angles)}
result = np.zeros((len(unique_angles), len(unique_beams), num_values), dtype=complex)

for i, angle in enumerate(unique_angles):
    for j, beam in enumerate(unique_beams):
        mask = (angles == angle) & (beams == beam)
        
        if np.sum(mask) == 1:
            result[i, j, :] = complex_data[mask, :].flatten()
        else:
            result[i, j, :] = complex_data[mask, :][0]

# SNR Calculation Function with visualization option
def calculate_snr(iq_data, visualize=False, title=""):
    """
    Calculate SNR of IQ data using power spectral density
    
    Args:
        iq_data: Complex IQ samples
        visualize: If True, create visualizations
        title: Title for the plot
        
    Returns:
        SNR value in dB
    """
    # Convert to magnitude
    magnitude = np.abs(iq_data)
    
    # Calculate power spectral density
    f, psd = welch(magnitude, fs=1.0, nperseg=min(256, len(magnitude)))
    
    # Find peak (signal)
    peak_idx = np.argmax(psd)
    signal_power = psd[peak_idx]
    
    # Calculate noise power (excluding the peak and adjacent bins)
    mask = np.ones_like(psd, dtype=bool)
    # Exclude peak and adjacent bins
    mask[max(0, peak_idx-2):min(len(mask), peak_idx+3)] = False
    noise_power = np.mean(psd[mask]) if np.any(mask) else 0.0
    
    # Calculate SNR
    if noise_power > 0:
        snr = 10 * np.log10(signal_power / noise_power)
    else:
        snr = float('inf')  # Avoid division by zero
    
    if visualize:
        fig, axs = plt.subplots(3, 1, figsize=(12, 12))
        
        # Plot 1: Raw IQ data
        axs[0].plot(np.real(iq_data), label='I (Real)')
        axs[0].plot(np.imag(iq_data), label='Q (Imag)')
        axs[0].set_title(f"{title} - Raw IQ Data")
        axs[0].set_xlabel('Sample')
        axs[0].set_ylabel('Amplitude')
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot 2: Magnitude
        axs[1].plot(magnitude)
        axs[1].set_title('Signal Magnitude')
        axs[1].set_xlabel('Sample')
        axs[1].set_ylabel('Magnitude')
        axs[1].grid(True)
        
        # Plot 3: Power Spectral Density with marked peak and noise floor
        axs[2].semilogy(f, psd)
        axs[2].axhline(y=noise_power, color='r', linestyle='--', label=f'Noise Floor: {10*np.log10(noise_power):.2f} dB')
        axs[2].plot(f[peak_idx], psd[peak_idx], 'go', markersize=10, label=f'Signal Peak: {10*np.log10(signal_power):.2f} dB')
        
        # Shade the areas used for noise calculation
        for i in range(len(mask)):
            if mask[i]:
                axs[2].axvspan(f[i]-0.5/(len(f)-1), f[i]+0.5/(len(f)-1), alpha=0.2, color='blue')
        
        axs[2].set_title(f'Power Spectral Density (SNR: {snr:.2f} dB)')
        axs[2].set_xlabel('Frequency')
        axs[2].set_ylabel('Power/Frequency (dB/Hz)')
        axs[2].legend()
        axs[2].grid(True)
        
        plt.tight_layout()
        plt.show()
        
    return snr

# Calculate SNR for each angle and beam combination
snr_matrix = np.zeros((len(unique_angles), len(unique_beams)))

for i in range(len(unique_angles)):
    for j in range(len(unique_beams)):
        iq_data = result[i, j, :]
        snr_matrix[i, j] = calculate_snr(iq_data)

# Display overall average SNR
average_snr = np.mean(snr_matrix)
print(f"Average SNR across all angles and beams: {average_snr:.2f} dB")

# Find maximum SNR and its location
max_snr = np.max(snr_matrix)
max_idx = np.unravel_index(np.argmax(snr_matrix), snr_matrix.shape)
max_angle = unique_angles[max_idx[0]]
max_beam = unique_beams[max_idx[1]]
print(f"Maximum SNR: {max_snr:.2f} dB at angle {max_angle}, beam {max_beam}")

# Plot the SNR heatmap
plt.figure(figsize=(12, 8))
plt.imshow(snr_matrix, aspect='auto', interpolation='none', 
           extent=[min(unique_beams), max(unique_beams), max(unique_angles), min(unique_angles)])
plt.colorbar(label='SNR (dB)')
plt.xlabel('Beam Index')
plt.ylabel('Angle (degrees)')
plt.title('SNR across angles and beams')
plt.plot(max_beam, max_angle, 'ro', markersize=10)  # Mark the maximum SNR point
plt.show()

# Now visualize the beam with maximum SNR
#i, j = max_idx
i = 10
j = 8
iq_data = result[i, j, :]
calculate_snr(iq_data, visualize=True, title=f"Beam {max_beam} at Angle {max_angle}°")

# Also visualize a beam with medium SNR for comparison
# Find a beam with SNR close to the median
median_snr = np.median(snr_matrix)
diff_from_median = np.abs(snr_matrix - median_snr)
med_idx = np.unravel_index(np.argmin(diff_from_median), diff_from_median.shape)
med_angle = unique_angles[med_idx[0]]
med_beam = unique_beams[med_idx[1]]
print(f"Medium SNR: {snr_matrix[med_idx]:.2f} dB at angle {med_angle}, beam {med_beam}")

iq_data = result[med_idx[0], med_idx[1], :]
calculate_snr(iq_data, visualize=True, title=f"Beam {med_beam} at Angle {med_angle}° (Medium SNR)")
# %%
