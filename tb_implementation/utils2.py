#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def add_noise(df, K):
    """Add Gaussian noise to beam gain data"""
    angles = df['Angle']
    power_levels = df['Power_Level']

    beam_columns = [col for col in df.columns if col.startswith('Beam_')]
    beam_data = df[beam_columns]
    noise_std_dev = K * beam_data.mean().mean() 

    noisy_beam_data = beam_data + np.random.normal(0, noise_std_dev, beam_data.shape)
    noisy_df = pd.concat([angles, power_levels, noisy_beam_data], axis=1)

    return noisy_df

def add_multipath(df, K):
    """Simulate multipath interference by adding clustered reflections"""

    num_clusters = np.random.randint(3, 6)  # Number of multipath clusters
    num_rays_per_cluster = np.random.randint(3, 6)  # Number of reflections per cluster

    angles = df['Angle'].to_numpy()
    beams = np.arange(0, 63)
    
    gain_profile = np.zeros((len(angles), len(beams)))

    for _ in range(num_clusters):
        cluster_center_angle = np.random.uniform(-45, 45)
        cluster_center_beam = np.random.randint(0, 63)

        for _ in range(num_rays_per_cluster):
            ray_angle = np.random.normal(cluster_center_angle, 5)  # Small angle spread
            ray_beam = np.random.normal(cluster_center_beam, 3)  # Small beam spread
            ray_amplitude = np.random.rayleigh(0.3)  # Rayleigh fading model
            ray_phase = np.random.uniform(0, 2*np.pi)

            for i, angle in enumerate(angles):
                for j, beam in enumerate(beams):
                    gain_profile[i, j] += ray_amplitude * np.cos(ray_phase) * np.exp(
                        -((angle - ray_angle) ** 2 / (2 * 2.5 ** 2))  # Spread in AoA domain
                        -((beam - ray_beam) ** 2 / (2 * 2.5 ** 2))  # Spread in beam domain
                    )

    # Normalize gains
    gains_existing = df.iloc[:, 2:].to_numpy(dtype=float)
    gains_norm = (gains_existing - np.min(gains_existing)) / (np.max(gains_existing) - np.min(gains_existing))
    
    gains_comb = gains_norm + (K * gain_profile)

    df_combined = pd.DataFrame(
        np.column_stack([df['Power_Level'].to_numpy(), df['Angle'].to_numpy(), gains_comb]),
        columns=df.columns
    )

    return df_combined

def add_multipath_rician(df, K):
    num_peaks = np.random.randint(3, 5)  # Number of scattered multipath peaks

    angles = np.linspace(-45, 45, 91)
    beams = np.arange(0, 63)

    gain_profile = np.zeros((91, 63))

    # Generate multipath components (Rayleigh-distributed)
    for _ in range(num_peaks):
        peak_angle = np.random.uniform(-45, 45)
        peak_beam = np.random.randint(0, 63)
        peak_gain = np.random.uniform(0, 1)

        angle_spread = np.random.uniform(1, 3)
        beam_spread = np.random.uniform(1, 3)

        for i, angle in enumerate(angles):
            for j, beam in enumerate(beams):
                gain_profile[i, j] += peak_gain * np.exp(
                    -((angle - peak_angle) ** 2 / (2 * angle_spread ** 2))
                    -((beam - peak_beam) ** 2 / (2 * beam_spread ** 2))
                )

    # Extract existing gain data (which includes LoS)
    gains_existing = df.iloc[:, 2:].to_numpy(dtype=float)

    #print(gains_existing.shape)

    # Normalize gains to [0,1]
    gains_norm = (gains_existing - np.min(gains_existing)) / (np.max(gains_existing) - np.min(gains_existing))

    # Apply Rician scaling: Existing gain (LoS) + scaled multipath (Rayleigh)
    gains_comb = np.sqrt(K / (K + 1)) * gains_norm + np.sqrt(1 / (K + 1)) * gain_profile

    # Create new DataFrame with updated gains
    df_combined = pd.DataFrame(
        np.column_stack([df['Angle'].to_numpy(), df['Power_Level'].to_numpy(), gains_comb]),
        columns=df.columns
    )

    return df_combined


def viz_df(df, figname):
    """Visualize the gain profile"""
    angs = df['Angle'].to_numpy()
    beams = np.arange(0, 63)
    gains = df.iloc[:, 2:].to_numpy()

    norm_gains = (gains - np.min(gains)) / (np.max(gains) - np.min(gains))

    X, Y = np.meshgrid(beams, angs)

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, norm_gains, cmap='jet', edgecolor='none')

    ax.set_xlabel("Beam")
    ax.set_ylabel("Angle")
    ax.set_zlabel("Gain")
    ax.set_title(figname)
    #plt.savefig(figname)


def get_snr(original_df, noisy_df):
    """Calculate the SNR of the noisy gain profile compared to the original gain profile."""
    # Extract the gain profiles from the dataframes
    original_gains = original_df.iloc[:, 2:].to_numpy()
    noisy_gains = noisy_df.iloc[:, 2:].to_numpy()

    # Calculate the signal power (average of squared original gains)
    signal_power = np.mean(original_gains**2)
    
    # Calculate the noise power (average of squared differences between noisy and original gains)
    noise_power = np.mean((noisy_gains - original_gains)**2)
    
    # Calculate the SNR in dB
    snr_db = 10 * np.log10(signal_power / noise_power)
    
    return snr_db

def adjust_noise_to_target_snr(original_df, target_snr, initial_K_multipath=5, initial_K_awgn=0.1, 
                               step_size_multipath=0.5, step_size_awgn=0.5, max_iterations=100):
    K_multipath = initial_K_multipath  # Multipath parameter (higher K means less multipath)
    K_awgn = initial_K_awgn  # AWGN noise parameter (higher K means more noise)
    iteration = 0
    
    while iteration < max_iterations:
        # Reset gain profile and apply multipath and AWGN separately
        #temp_df = add_multipath_rician(original_df, K_multipath)
        noisy_df = add_noise(original_df, K_awgn)

        # Compute SNR
        current_snr = get_snr(original_df, noisy_df)

        #print(f"Iteration {iteration}: K_multipath={K_multipath:.2f}, K_awgn={K_awgn:.2f}, SNR={current_snr:.2f} dB")

        # Check if target SNR is reached
        if current_snr <= target_snr:
            return noisy_df, current_snr

        # Adjust multipath and AWGN parameters
        K_multipath = max(0.1, K_multipath - step_size_multipath)  # Reduce multipath K (increase multipath effect)
        K_awgn += step_size_awgn  # Increase AWGN noise

        iteration += 1

    #print("Warning: Maximum iterations reached without achieving target SNR.")
    return noisy_df, current_snr

def run_test():
    df = pd.read_csv('../data_processing/train_gain_prof.csv')
    viz_df(df, 'original_gain_profile.png')

    target_snr = 18  # Target SNR in dB
    noisy_df, final_snr = adjust_noise_to_target_snr(df, target_snr)

    print(f"Final achieved SNR: {final_snr:.2f} dB")

    viz_df(noisy_df, 'noisy_df')

    noisy_df.to_csv('../data/spacial_profiles/mid_high_snr/gp_' + f"{final_snr:.2f}" + '.csv', index=False)

    #mpdf = add_multipath_rician(df, 10)
    #viz_df(mpdf, 'multipath_gain_profile.png')

    #ndf = add_noise(mpdf, 0.1)
    #viz_df(ndf, 'noisy_gain_profile.png')

    #print("SNR:")
    #snr = get_snr(df, ndf)
    #print(snr)

#run_test()

# %%
