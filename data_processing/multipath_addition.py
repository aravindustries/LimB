#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_angles = 91  # Angles from -45 to 45 degrees
num_beams = 63  # Beams from 0 to 62
num_peaks = np.random.randint(3, 5)  # Random number of peaks

angles = np.linspace(-45, 45, num_angles)
beams = np.arange(0, num_beams)

# Generate gain profiles with multiple peaks
gain_profiles = np.zeros((num_angles, num_beams))

for _ in range(num_peaks):
    peak_angle = np.random.uniform(-45, 45)  # Random peak angle
    peak_beam = np.random.randint(0, num_beams)  # Random peak beam
    peak_gain = np.random.uniform(0, 1)  # Peak gain value

    # Create smooth Gaussian-like peaks
    angle_spread = np.random.uniform(2, 3)  # Width of peak across angles
    beam_spread = np.random.uniform(2, 3)  # Width of peak across beams

    for i, angle in enumerate(angles):
        for j, beam in enumerate(beams):
            gain_profiles[i, j] += peak_gain * np.exp(
                -((angle - peak_angle) ** 2 / (2 * angle_spread ** 2)) 
                -((beam - peak_beam) ** 2 / (2 * beam_spread ** 2))
            )

# Normalize gains between 0 and 1
gain_profiles = (gain_profiles - np.min(gain_profiles)) / (np.max(gain_profiles) - np.min(gain_profiles))

# Set Power_Level to all zeros
power_levels = np.zeros(num_angles)

# Create DataFrame with proper headers
columns = ["Power_Level", "Angle"] + [f"Beam_{i}" for i in range(num_beams)]
df_generated = pd.DataFrame(np.column_stack([power_levels, angles, gain_profiles]), columns=columns)

# Save to CSV
generated_file = "../data_processing/multipath.csv"
df_generated.to_csv(generated_file, index=False)
print(f"Generated random data saved as: {generated_file}")

# Load existing and generated data
existing_file = "../data_processing/power_800_data.csv"  # Replace with the path to your existing data
df_existing = pd.read_csv(existing_file)
df_generated = pd.read_csv(generated_file)

# Ensure both files have the same structure
if not df_existing.columns.equals(df_generated.columns):
    raise ValueError("CSV files have different structures. Ensure they match before processing.")

# Extract angle and power level columns
angles = df_existing["Angle"].to_numpy()
power_levels = df_existing["Power_Level"].to_numpy()

# Extract beam gain values
gains_existing = df_existing.iloc[:, 2:].to_numpy(dtype=float)
gains_generated = df_generated.iloc[:, 2:].to_numpy(dtype=float)

# Normalize the existing gains (between 0 and 1)
gains_existing_normalized = (gains_existing - np.min(gains_existing)) / (np.max(gains_existing) - np.min(gains_existing))

# Normalize the generated gains (between 0 and 1)
gains_generated_normalized = (gains_generated - np.min(gains_generated)) / (np.max(gains_generated) - np.min(gains_generated))

# Scale the normalized generated gains and add them to the normalized existing gains
K = 0.8  # Scaling factor for the generated data
gains_combined = gains_existing_normalized + (K * gains_generated_normalized)

# Debugging: Check the combined gains for peaks
print("Sample of Combined Gains (Existing + Generated):\n", gains_combined[:5, :5])

# Create new DataFrame with mixed gains (ensure correct column order)
df_combined = pd.DataFrame(
    np.column_stack([power_levels, angles, gains_combined]), 
    columns=df_existing.columns
)

# Save to new CSV
output_file = "../data_processing/multipath_mixed.csv"
df_combined.to_csv(output_file, index=False)
print(f"Combined file saved as: {output_file}")

# Function to plot 3D gain profiles
def plot_gain_profile(df, title):
    angles = df["Angle"].to_numpy()
    beams = np.arange(df.shape[1] - 2)  # Exclude Angle and Power_Level columns
    gains = df.iloc[:, 2:].to_numpy(dtype=float)

    # Normalize gains for visualization
    norm_gains = (gains - np.min(gains)) / (np.max(gains) - np.min(gains))

    X, Y = np.meshgrid(beams, angles)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, norm_gains, cmap='jet', edgecolor='none')

    ax.set_xlabel("Beam")
    ax.set_ylabel("Angle")
    ax.set_zlabel("Gain")
    ax.set_title(title)

    plt.show()

# Plot generated gain profile
plot_gain_profile(df_generated, "Generated Gain Profile")

# Plot existing gain profile
plot_gain_profile(df_existing, "Existing Gain Profile")

# Plot new gain profile with mixed data
plot_gain_profile(df_combined, "New Gain Profile (Mixed)")

# %%
