import matplotlib.pyplot as plt
import numpy as np

sample_rate = 1e6
N = 10000
d = 0.55


def generate_data(
    num_samples,
    Nr=16,  # Default for Sivers
    N_snapshots=512,
    snr_range=(-20, 10),
    theta_range=(-32, 32),
    num_of_paths_range=(1, 1),
):
    X = np.zeros((num_samples, 2 * Nr, N_snapshots), dtype=np.float32)
    y = np.zeros(num_samples)

    num_of_paths = np.random.randint(num_of_paths_range[0], num_of_paths_range[1]+1)
    emph = 0.5 ** np.arange(num_of_paths)

    for j in range(num_of_paths):
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
            r = (s @ tx) * emph[j]
            r *= np.exp(-2j * np.pi * np.random.uniform())

            # Add AWGN noise based on SNR
            snr_db = np.random.uniform(*snr_range)
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

            X[i, :Nr, :] += I_components
            X[i, Nr:, :] += Q_components

    return X, y


if __name__ == "__main__":
    Nr=16

    X, _ = generate_data(1, Nr=Nr, N_snapshots=512, snr_range=(0, 0), num_of_paths_range=(2, 2))
    X = X[0, :Nr] + 1j * X[0, Nr:] # combine I and Q components into a single complex array

    theta_scan = np.linspace(-1*np.pi/2, np.pi/2, 1000)
    results = []
    for theta_i in theta_scan:
        w = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta_i)) # Conventional, aka delay-and-sum, beamformer
        # breakpoint()
        X_weighted = w.conj().T @ X # apply our weights. remember X is 3x10000
        results.append(10*np.log10(np.var(X_weighted))) # power in signal, in dB so its easier to see small and large lobes at the same time
    results -= np.max(results) # normalize (optional)

    plt.plot(theta_scan*180/np.pi, results) # lets plot angle in degrees
    plt.xlabel("Theta [Degrees]")
    plt.ylabel("DOA Metric")
    plt.grid()
    plt.show()

