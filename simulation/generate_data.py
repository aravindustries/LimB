import numpy as np
import matplotlib.pyplot as plt

sample_rate = 1e6
N = 10000
d = 0.55


def generate_data(
    num_samples,
    Nr=4,  # Default for Sivers
    N_snapshots=512,
    snr_range=(-20, 10),
    theta_range=(-32, 32),
    num_of_thetas_range=(1, 2),
):
    X = np.zeros((num_samples, 2 * Nr, N_snapshots), dtype=np.float32)
    y = np.zeros(num_samples)

    for i in range(num_samples):
        # breakpoint()
        num_of_thetas = np.random.randint(*num_of_thetas_range)
        theta_degrees = np.random.uniform(*theta_range, size=int(num_of_thetas))
        # print(theta_degrees)
        # breakpoint()
        y[i] = np.round(theta_degrees[0]) - 32

        theta = theta_degrees / 180 * np.pi
        # print(theta)
        # breakpoint()
        s = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta.reshape(-1, 1))).T
        # print(s.shape)

        t = np.arange(N_snapshots) / sample_rate
        f_tone = 0.02e6

        

        tx = np.exp(2j * np.pi * f_tone * t).reshape(1, -1)

        
        r_all_paths = np.zeros([num_of_thetas, Nr, N_snapshots])
        
        for j in range(num_of_thetas):
            # breakpoint()
            r_all_paths[j] = s[:, j].reshape(-1, 1) @ tx

        r = np.zeros([Nr, N_snapshots], dtype=np.complex128)
        
        for j in range(num_of_thetas):
            r += r_all_paths[j] * (0.2 ** j)
        # print(r.shape)

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

        X[i, :Nr] = I_components
        X[i, Nr:] = Q_components

    return X, y


# if __name__ == "__main__": 
#     Nr = 16
#     X, _ = generate_data(1, Nr=Nr, num_of_thetas=(2,3), theta_range=(-40, 40))

#     X = X[0, :Nr] + 1j * X[0, Nr:]

#     alphas = np.arange(-40, 41) * np.pi / 180
#     alpha_matrix = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(alphas.reshape(-1, 1)))

#     signals = alpha_matrix @ X.conj()
#     profile = np.var(signals, axis=1)

#     plt.plot(profile)
#     plt.show()
