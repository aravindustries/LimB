import numpy as np

sample_rate = 1e6
N = 10000
d = 0.55


def generate_data(
    num_samples,
    Nr=16,  # Default for Sivers
    N_snapshots=512,
    snr_range=(-20, 10),
    k_factor_range=(2, 10),
    theta_range=(-32, 32),
):
    X = np.zeros((num_samples, 2 * Nr, N_snapshots), dtype=np.float32)
    y = np.zeros(num_samples)

    for i in range(num_samples):
        theta_degrees = np.random.uniform(*theta_range)
        y[i] = np.round(theta_degrees) + 32

        theta = theta_degrees / 180 * np.pi

        # Line of sight component
        s = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta)).reshape(-1, 1)

        t = np.arange(N_snapshots) / sample_rate
        f_tone = 0.02e6
        tx = np.exp(2j * np.pi * f_tone * t).reshape(1, -1)

        # LOS signal component
        r_los = s @ tx

        # Generate K-factor for Rician fading (ratio of LOS to scattered power)
        k_factor = np.random.uniform(k_factor_range[0], k_factor_range[1])

        # Calculate power ratio
        los_power = k_factor / (k_factor + 1)
        scatter_power = 1 / (k_factor + 1)

        # Generate multipath components (scattered paths)
        # This creates Rayleigh fading for the scattered component
        h_scatter = np.sqrt(scatter_power / 2) * (
            np.random.randn(Nr, 1) + 1j * np.random.randn(Nr, 1)
        )
        r_scatter = h_scatter @ tx

        # Scale LOS component
        r_los = np.sqrt(los_power) * r_los

        # Combine LOS and scattered components to create Rician fading
        r = r_los + r_scatter

        # Add AWGN noise based on SNR
        snr_db = np.random.uniform(snr_range[0], snr_range[1])
        snr_linear = 10 ** (snr_db / 10)

        signal_power = np.mean(np.abs(r) ** 2)
        noise_power = signal_power / snr_linear
        noise_std = np.sqrt(noise_power / 2)  # divide by 2 for complex noise

        noise = noise_std * (
            np.random.randn(Nr, N_snapshots) + 1j * np.random.randn(Nr, N_snapshots)
        )

        r = r + noise

        I_components = np.real(r)
        Q_components = np.imag(r)

        X[i, :Nr, :] = I_components
        X[i, Nr:, :] = Q_components

    return X, y
