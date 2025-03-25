#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_angles = 91  # Angles from -45 to 45 degrees
num_beams = 63  # Beams from 0 to 62
num_peaks = np.random.randint(3, 6)  # Random number of peaks

angles = np.linspace(-45, 45, num_angles)
beams = np.arange(0, num_beams)

# Generate gain profiles with multiple peaks
gain_profiles = np.zeros((num_angles, num_beams))

for _ in range(num_peaks):
    peak_angle = np.random.uniform(-45, 45)  # Random peak angle
    peak_beam = np.random.randint(0, num_beams)  # Random peak beam
    peak_gain = np.random.uniform(0.1, 0.6)  # Peak gain value

    # Create smooth Gaussian-like peaks
    angle_spread = np.random.uniform(3, 5)  # Width of peak across angles
    beam_spread = np.random.uniform(1, 3)  # Width of peak across beams

    for i, angle in enumerate(angles):
        for j, beam in enumerate(beams):
            gain_profiles[i, j] += peak_gain * np.exp(
                -((angle - peak_angle) ** 2 / (2 * angle_spread ** 2)) 
                -((beam - peak_beam) ** 2 / (2 * beam_spread ** 2))
            )

# Normalize gains between 0 and 1
gain_profiles = (gain_profiles - np.min(gain_profiles)) / (np.max(gain_profiles) - np.min(gain_profiles))

# Add angle and random power levels
tx_powers = np.random.uniform(-20, 5, num_angles)  # Simulated transmission power levels

# Create DataFrame
df = pd.DataFrame(np.column_stack([angles, tx_powers, gain_profiles]), columns=["Angle", "Power_Level"] + [f"Beam_{i}" for i in range(num_beams)])

# Save to CSV
csv_filename = "../data_processing/multipath.csv"
df.to_csv(csv_filename, index=False, header=False)

# Function to visualize the gain profile
def viz_gain_prof(csv_file):
    df = pd.read_csv(csv_file, header=None)

    angs = df[0].to_numpy()
    tx_powers = df[1].to_numpy()
    beams = np.arange(0, 63)
    gains = df.iloc[:, 2:65].to_numpy()

    norm_gains = (gains - np.min(gains)) / (np.max(gains) - np.min(gains))

    X, Y = np.meshgrid(beams, angs)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, norm_gains, cmap="jet", edgecolor="none")

    ax.set_xlabel("Beam")
    ax.set_ylabel("Angle")
    ax.set_zlabel("Gain")
    ax.set_title("3D Gain Plot: " + csv_file)

    plt.show()

# Visualize the new gain profile
viz_gain_prof(csv_filename)






# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_angles = 91  # Angles from -45 to 45 degrees
num_beams = 63  # Beams from 0 to 62
num_peaks = np.random.randint(1, 5)  # Random number of peaks

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
df = pd.DataFrame(np.column_stack([power_levels, angles, gain_profiles]), columns=columns)

# Save to CSV
csv_filename = "../data_processing/multipath.csv"
df.to_csv(csv_filename, index=False)

# Function to visualize the gain profile
def viz_gain_prof(csv_file):
    df = pd.read_csv(csv_file)

    angs = df["Angle"].to_numpy()
    tx_powers = df["Power_Level"].to_numpy()
    beams = np.arange(0, 63)
    gains = df.iloc[:, 2:65].to_numpy()

    norm_gains = (gains - np.min(gains)) / (np.max(gains) - np.min(gains))

    X, Y = np.meshgrid(beams, angs)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, norm_gains, cmap="jet", edgecolor="none")

    ax.set_xlabel("Beam")
    ax.set_ylabel("Angle")
    ax.set_zlabel("Gain")
    ax.set_title("3D Gain Plot: " + csv_file)

    plt.show()

# Visualize the new gain profile
viz_gain_prof(csv_filename)





# %%
import pandas as pd
import numpy as np

# Define file paths
generated_file = "../data_processing/multipath.csv"  # Replace with actual file path
existing_file = "../data_processing/power_800_data.csv"  # Replace with actual file path
output_file = "../data_processing/multipath_mixed.csv"  # New output file

# Scaling factor for beam values
K = 10  # Change this as needed

# Load existing and generated data
df_existing = pd.read_csv(existing_file)
df_generated = pd.read_csv(generated_file)

# Ensure both files have the same structure
if not df_existing.columns.equals(df_generated.columns):
    raise ValueError("CSV files have different structures. Ensure they match before processing.")

# Extract angle and power level columns (keep unchanged)
angles = df_existing["Angle"]
power_levels = df_existing["Power_Level"]

# Extract beam gain values
gains_existing = df_existing.iloc[:, 2:].to_numpy()
gains_generated = df_generated.iloc[:, 2:].to_numpy()

# Scale the new generated gains and add to existing gains element-wise
gains_combined = gains_existing + (K * gains_generated)

# Create new DataFrame
df_combined = pd.DataFrame(np.column_stack([power_levels, angles, gains_combined]), columns=df_existing.columns)

# Save to new CSV
df_combined.to_csv(output_file, index=False)

print(f"Combined file saved as: {output_file}")





# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_gain_profile(csv_file, title):
    df = pd.read_csv(csv_file)

    angles = df["Angle"].to_numpy()
    beams = np.arange(0, 63)
    gains = df.iloc[:, 2:].to_numpy()

    # Normalize gains for better visualization
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


# File paths
existing_file = "../data_processing/power_800_data.csv"
output_file = "../data_processing/multipath_mixed.csv"

# Plot existing gain profile
plot_gain_profile(existing_file, "Existing Gain Profile")

# Plot new gain profile with generated data mixed in
plot_gain_profile(output_file, "New Gain Profile (Mixed)")












# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define file paths
generated_file = "../data_processing/multipath.csv"
existing_file = "../data_processing/power_800_data.csv"
output_file = "../data_processing/multipath_mixed.csv"

# Scaling factor for beam values
K = 10  # Adjust as needed

# Load existing and generated data
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

# Scale the generated gains and add them to the existing gains
gains_combined = gains_existing + (K * gains_generated)

# Debugging: Print samples to verify changes
print("Original Gains Sample:\n", gains_existing[:5, :5])
print("Generated Gains Sample:\n", gains_generated[:5, :5])
print("Combined Gains Sample:\n", gains_combined[:5, :5])

# Create new DataFrame with mixed gains
df_combined = pd.DataFrame(
    np.column_stack([power_levels, angles, gains_combined]), 
    columns=df_existing.columns
)

# Save to new CSV
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

# Load and plot the existing gain profile
df_existing = pd.read_csv(existing_file)
plot_gain_profile(df_existing, "Existing Gain Profile")

# Load and plot the new gain profile with mixed data
df_combined = pd.read_csv(output_file)
plot_gain_profile(df_combined, "New Gain Profile (Mixed)")






# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define file paths
generated_file = "../data_processing/multipath.csv"  # Generated data
existing_file = "../data_processing/power_800_data.csv"  # Existing data
output_file = "../data_processing/multipath_mixed.csv"  # New output file

# Scaling factor for beam values
K = 0.8  # Adjust as needed

# Function to generate random data with peaks (for testing purposes)
def generate_random_data():
    angles = np.linspace(-45, 45, 91)  # Angles from -45 to 45 degrees
    power_level = np.zeros_like(angles)  # Power level for all angles is zero
    
    # Generate random peaks in the beam gain profile
    num_peaks = 10  # Adjust the number of peaks
    beam_values = np.zeros((angles.shape[0], 63))  # 63 beam values
    
    for _ in range(num_peaks):
        peak_angle_idx = np.random.randint(0, len(angles))  # Random angle index
        peak_beam_idx = np.random.randint(0, 63)  # Random beam index
        
        # Set a random peak value at that position
        peak_value = np.random.uniform(5, 30)  # Peak value between 5 and 30 dB
        beam_values[peak_angle_idx, peak_beam_idx] = peak_value

    # Create a DataFrame with random data and the correct structure
    df_generated = pd.DataFrame(
        np.column_stack([power_level, angles, beam_values]), 
        columns=["Power_Level", "Angle"] + [f"Beam_{i}" for i in range(63)]
    )
    
    return df_generated

# Generate the random data (for testing purposes)
df_generated = generate_random_data()

# Save the generated data to a CSV file
df_generated.to_csv(generated_file, index=False)
print(f"Generated random data saved as: {generated_file}")

# Load existing and generated data
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
gains_combined = gains_existing_normalized + (K * gains_generated_normalized)

# Debugging: Check the combined gains for peaks
print("Sample of Combined Gains (Existing + Generated):\n", gains_combined[:5, :5])

# Create new DataFrame with mixed gains (ensure correct column order)
df_combined = pd.DataFrame(
    np.column_stack([power_levels, angles, gains_combined]), 
    columns=df_existing.columns
)

# Save to new CSV
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
