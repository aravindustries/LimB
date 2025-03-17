import numpy as np


class Music:
    def __init__(self, d):
        self.d = d

    def evaluate(self, X, y, Nr=16):
        I_components = X[:, :Nr]
        Q_components = X[:, Nr:]
        r = I_components + 1j * Q_components

        Nr = r.shape[1]

        # Calculate covariance matrix
        R = r @ r.conj().transpose(0,2,1)

        # Eigenvalue decomposition
        w, v = np.linalg.eig(R)

        # Sort eigenvalues and eigenvectors
        eig_val_order = np.argsort(np.abs(w))
        v_sorted = np.take_along_axis(v, eig_val_order[:, np.newaxis, :], axis=2)

        # Noise subspace
        V = v_sorted[:, :, :Nr - 1]
        proj = V @ V.conj().transpose(0, 2, 1)

        # Scan angles
        theta_scan = np.linspace(-np.pi/2, np.pi/2, 181)  # -90 to 90 degrees

        s = np.exp(-2j * np.pi * self.d * np.arange(Nr)[:, np.newaxis] * np.sin(theta_scan))
        results = np.einsum('ij,njk,ki->ni', s.conj().T, proj, s)
        results = np.abs(results)

        results = np.array(results)
        indices = np.argmin(results, axis=1)
        
        true_angles = y.astype(int) + 90 - 32
        
        accuracy = (indices == true_angles).mean()
        mae = np.abs(indices - true_angles).mean()

        return accuracy, mae
