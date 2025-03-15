import numpy as np


class Music:
    def __init__(self, d):
        self.d = d

    def evaluate(self, X, y, Nr):
        correct_count = 0
        total_error = 0

        for i in range(len(X)):
            # Reconstruct complex signal from I/Q components
            I_components = X[i, :Nr, :]
            Q_components = X[i, Nr:, :]
            r_complex = I_components + 1j * Q_components

            # Run MUSIC algorithm
            _, _, theta_music = self.music_algorithm(r_complex, num_sources=1)

            # Get true angle (convert from class index back to angle)
            true_angle = int(y[i]) - 32

            # Check if MUSIC found any peaks
            if len(theta_music) > 0:
                theta_music_val = theta_music[0]
            else:
                theta_music_val = 0  # Default if MUSIC fails to find a peak

            # Calculate error
            error = np.abs(theta_music_val - true_angle)
            total_error += error

            # Count as correct if error is less than 0.5 degrees
            if error < 0.5:
                correct_count += 1

        # Calculate accuracy and mean absolute error
        accuracy = (correct_count / len(X)) * 100
        mae = total_error / len(X)

        return accuracy, mae

    def music_algorithm(self, r, num_sources=1):
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
            s = np.exp(-2j * np.pi * self.d * np.arange(Nr) * np.sin(theta_i)).reshape(
                -1, 1
            )
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
