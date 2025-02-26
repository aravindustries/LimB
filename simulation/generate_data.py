import numpy as np

sample_rate = 1e6
N = 10000
d = 0.5


def generate_data(num_samples, Nr=8, N_snapshots=512, snr_range=(-10, 10)):
    """Generate training data for DOA estimation using IQ components."""
    X = np.zeros((num_samples, 2 * Nr, N_snapshots), dtype=np.float32)
    y = np.zeros(num_samples)

    for i in range(num_samples):
        theta_degrees = np.random.uniform(-32, 32)
        y[i] = np.round(theta_degrees) + 32

        theta = theta_degrees / 180 * np.pi

        s = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta)).reshape(-1, 1)

        t = np.arange(N_snapshots) / sample_rate
        f_tone = 0.02e6
        tx = np.exp(2j * np.pi * f_tone * t).reshape(1, -1)

        r = s @ tx

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
