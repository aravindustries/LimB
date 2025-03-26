#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('power_800_data.csv')

# Extract the angle and power level columns
angles = df['Angle']
power_levels = df['Power_Level']

# Extract the beam columns
beam_columns = [col for col in df.columns if col.startswith('Beam_')]
beam_data = df[beam_columns]

# Define the standard deviation for the Gaussian noise
# Adjust the noise level as needed
noise_std_dev = 0.05 * beam_data.mean().mean()  # 5% of the mean gain value

# Apply Gaussian noise to the beam data
noisy_beam_data = beam_data + np.random.normal(0, noise_std_dev, beam_data.shape)

# Combine the noisy data with the angle and power level columns
noisy_df = pd.concat([angles, power_levels, noisy_beam_data], axis=1)

# Save the noisy data to a new CSV file
# Replace 'noisy_file.csv' with your desired output file name
noisy_df.to_csv('noisy_file.csv', index=False)

# Plot the original and noisy gain profiles for a specific angle
# Replace target_angle with the angle you want to analyze
target_angle = -45.0
angle_index = angles[angles == target_angle].index[0]

plt.figure(figsize=(12, 6))
plt.plot(beam_columns, beam_data.iloc[angle_index], label='Original', marker='o')
plt.plot(beam_columns, noisy_beam_data.iloc[angle_index], label='Noisy', marker='x')
plt.xlabel('Beam')
plt.ylabel('Gain')
plt.title(f'Gain Profiles at Angle {target_angle}°')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('power_800_data.csv')

# Extract the angle and power level columns
angles = df['Angle']
power_levels = df['Power_Level']

# Extract the beam columns
beam_columns = [col for col in df.columns if col.startswith('Beam_')]
beam_data = df[beam_columns]

# Define the standard deviation for the Gaussian noise
noise_std_dev = 0.05 * beam_data.mean().mean()  # 5% of the mean gain value

# Generate Gaussian noise
noise = np.random.normal(0, noise_std_dev, beam_data.shape)

# Apply Gaussian noise to the beam data
noisy_beam_data = beam_data + noise

# Compute noise statistics
mean_noise = np.mean(noise)
std_noise = np.std(noise)
snr = 10 * np.log10(np.mean(beam_data**2) / np.mean(noise**2))
mae = np.mean(np.abs(noisy_beam_data - beam_data))
rmse = np.sqrt(np.mean((noisy_beam_data - beam_data) ** 2))

# Print noise metrics
print(f"Mean Noise: {mean_noise:.2f}")
print(f"Standard Deviation of Noise: {std_noise:.2f}")
print(f"SNR (dB): {snr:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Square Error (RMSE): {rmse:.2f}")

# Save the noisy data to a new CSV file
noisy_df = pd.concat([angles, power_levels, noisy_beam_data], axis=1)
noisy_df.to_csv('noisy_file.csv', index=False)

# Plot original and noisy gain profiles for a specific angle
target_angle = 12.0
angle_index = angles[angles == target_angle].index[0]

plt.figure(figsize=(12, 6))
plt.plot(beam_columns, beam_data.iloc[angle_index], label='Original', marker='o')
plt.plot(beam_columns, noisy_beam_data.iloc[angle_index], label='Noisy', marker='x')
plt.xlabel('Beam')
plt.ylabel('Gain')
plt.title(f'Gain Profiles at Angle {target_angle}°')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

# %%
