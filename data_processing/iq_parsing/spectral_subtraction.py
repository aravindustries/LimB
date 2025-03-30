#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

# Load and process the data
df = pd.read_csv('../../../beamtracking17/combined_beam_iq.csv', header=None)
df.drop(df.columns[1], axis=1, inplace=True)
angles = df.iloc[:, 0].values
beams = df.iloc[:, 1].values
complex_data = df.iloc[:, 2:].values

# Assuming data is already in complex form or needs to be processed differently
# Let's not try to convert it to complex since that's causing the error

unique_angles = np.unique(angles)
unique_beams = np.unique(beams)
num_values = complex_data.shape[1]

# Organize data by angle and beam
result = np.zeros((len(unique_angles), len(unique_beams), num_values), dtype=complex)

def get_psd(iq_data, fs=1000):  # Assuming 1000 Hz sampling rate for better visualization
    f, psd = welch(iq_data, fs=fs, nperseg=min(256, len(iq_data)))
    return f, psd

# Store all IQ samples for processing
all_iq_samples = []

for i, angle in enumerate(unique_angles):
    for j, beam in enumerate(unique_beams):
        mask = (angles == angle) & (beams == beam)
        
        if np.sum(mask) == 1:
            # Check the first few values to see if they're already complex
            sample_data = complex_data[mask, :].flatten()
            result[i, j, :] = sample_data
        else:
            # If multiple matches, take the first one
            result[i, j, :] = complex_data[mask, :][0]
        
        # Collect all IQ samples
        all_iq_samples.append(result[i, j, :])

# Convert to array for easier processing
all_iq_samples = np.array(all_iq_samples)

# Calculate average IQ sample across all beams and angles
avg_iq_sample = np.mean(all_iq_samples, axis=0)

# Print some diagnostic information
print(f"Shape of avg_iq_sample: {avg_iq_sample.shape}")
print(f"First few values of avg_iq_sample: {avg_iq_sample[:5]}")
print(f"Data type: {avg_iq_sample.dtype}")

# Calculate PSD of the average IQ sample
fs = 1000  # Sampling frequency in Hz
f, avg_psd = get_psd(avg_iq_sample, fs)

# Plot the original average PSD
plt.figure(figsize=(10, 6))
plt.semilogy(f, avg_psd)
plt.title('Original Average PSD')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')
plt.grid(True)
plt.show()

# Plot the original time-domain signal
plt.figure(figsize=(10, 6))
plt.plot(np.real(avg_iq_sample))
plt.title('Original Time-Domain Signal (Real Part)')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# Estimate the signal and noise regions
# Assuming signal is concentrated around 0.4*fs Hz
signal_region_min = 0.3 * fs
signal_region_max = 0.5 * fs

# Find the appropriate indices in the frequency array
signal_mask = (f >= signal_region_min/fs) & (f <= signal_region_max/fs)
noise_mask = ~signal_mask

# Estimate noise power
noise_power = np.mean(avg_psd[noise_mask])
print(f"Estimated noise power: {noise_power}")

# Perform spectral subtraction
denoised_psd = np.copy(avg_psd)
denoised_psd[noise_mask] = np.maximum(denoised_psd[noise_mask] - noise_power * 0.9, 0.01 * noise_power)  # Subtract 90% of noise, leave some floor

# Plot the denoised PSD
plt.figure(figsize=(10, 6))
plt.semilogy(f, avg_psd, label='Original')
plt.semilogy(f, denoised_psd, label='Denoised')
plt.title('Original vs Denoised PSD')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')
plt.legend()
plt.grid(True)
plt.show()

# Signal reconstruction using Wiener filtering approach
wiener_filter = np.zeros_like(denoised_psd)
nonzero_mask = avg_psd > 1e-10  # Avoid division by zero
wiener_filter[nonzero_mask] = denoised_psd[nonzero_mask] / avg_psd[nonzero_mask]

# Apply the filter in frequency domain
fft_signal = np.fft.rfft(avg_iq_sample)
# Make sure the filter is the right size
filter_resized = np.interp(
    np.linspace(0, 1, len(fft_signal)), 
    np.linspace(0, 1, len(wiener_filter)), 
    wiener_filter
)
fft_denoised = fft_signal * filter_resized
denoised_signal = np.fft.irfft(fft_denoised, len(avg_iq_sample))

# Plot the denoised time-domain signal
plt.figure(figsize=(10, 6))
plt.plot(np.real(avg_iq_sample), alpha=0.7, label='Original')
plt.plot(np.real(denoised_signal), label='Denoised')
plt.title('Original vs Denoised Time-Domain Signal (Real Part)')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()

# Alternative approach: Simple bandpass filtering
from scipy.signal import butter, filtfilt

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Apply bandpass filter
bandpass_filtered = np.zeros_like(avg_iq_sample, dtype=complex)
bandpass_filtered.real = bandpass_filter(np.real(avg_iq_sample), signal_region_min, signal_region_max, fs)
bandpass_filtered.imag = bandpass_filter(np.imag(avg_iq_sample), signal_region_min, signal_region_max, fs)

# Calculate PSD of the bandpass filtered signal
f_bp, bp_psd = get_psd(bandpass_filtered, fs)

# Plot the bandpass filtered PSD
plt.figure(figsize=(10, 6))
plt.semilogy(f, avg_psd, label='Original')
plt.semilogy(f_bp, bp_psd, label='Bandpass Filtered')
plt.title('Original vs Bandpass Filtered PSD')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')
plt.legend()
plt.grid(True)
plt.show()

# Plot the bandpass filtered time-domain signal
plt.figure(figsize=(10, 6))
plt.plot(np.real(avg_iq_sample), alpha=0.7, label='Original')
plt.plot(np.real(bandpass_filtered), label='Bandpass Filtered')
plt.title('Original vs Bandpass Filtered Time-Domain Signal (Real Part)')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()