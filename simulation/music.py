import numpy as np

sample_rate = 1e6
N = 10000
d = 0.5


def music_algorithm(r, num_sources=1):
    Nr = r.shape[0]

    # Calculate covariance matrix
    R = r @ r.conj().T

    # Eigenvalue decomposition
    w, v = np.linalg.eig(R)

    # Sort eigenvalues and eigenvectors
    eig_val_order = np.argsort(np.abs(w))
    v = v[:, eig_val_order]

    # Noise subspace
    V = v[
        :, : Nr - num_sources
    ]  # Noise subspace is the eigenvectors with smallest eigenvalues

    # Scan angles
    theta_scan = np.linspace(-np.pi / 2, np.pi / 2, 181)  # -90 to 90 degrees
    results = []

    for theta_i in theta_scan:
        s = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta_i)).reshape(-1, 1)
        metric = 1 / (s.conj().T @ V @ V.conj().T @ s)  # The main MUSIC equation
        metric = np.abs(metric.squeeze())
        results.append(metric)

    results = np.array(results)
    results_db = 10 * np.log10(results / np.max(results))

    # Find peaks in the spectrum
    peaks = []
    for i in range(1, len(results_db) - 1):
        if results_db[i] > results_db[i - 1] and results_db[i] > results_db[i + 1]:
            peaks.append((theta_scan[i] * 180 / np.pi, results_db[i]))

    # Sort peaks by amplitude
    peaks.sort(key=lambda x: x[1], reverse=True)

    # Return top num_sources peaks as DOA estimates
    doa_estimates = [peak[0] for peak in peaks[:num_sources]]

    return theta_scan * 180 / np.pi, results_db, np.array(doa_estimates)
