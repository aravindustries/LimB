import matplotlib as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

from cnn import build_resnet_model, evaluate_model
from generate_data import generate_data
from music import music_algorithm

sample_rate = 1e6
N = 10000
d = 0.5


# Function to compare CNN and MUSIC performance
def compare_cnn_music(cnn_model, snr_levels=[-10, 0, 10], num_samples=100):
    Nr = 8
    N_snapshots = 512

    results = []

    for snr in snr_levels:
        cnn_rmse = 0
        music_rmse = 0

        for _ in range(num_samples):
            # Generate a test sample
            theta_true = np.random.uniform(-80, 80)  # Avoid edge cases

            # Generate received signal
            theta_rad = theta_true * np.pi / 180
            s = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta_rad)).reshape(
                -1, 1
            )
            t = np.arange(N_snapshots) / sample_rate
            f_tone = 0.02e6
            tx = np.exp(2j * np.pi * f_tone * t).reshape(1, -1)
            r = s @ tx

            # Add noise
            snr_linear = 10 ** (snr / 10)
            signal_power = np.mean(np.abs(r) ** 2)
            noise_power = signal_power / snr_linear
            noise_std = np.sqrt(noise_power / 2)
            noise = noise_std * (
                np.random.randn(Nr, N_snapshots) + 1j * np.random.randn(Nr, N_snapshots)
            )
            r_noisy = r + noise

            # CNN estimation
            I_components = np.real(r_noisy)
            Q_components = np.imag(r_noisy)
            X = np.zeros((1, 2 * Nr, N_snapshots), dtype=np.float32)
            X[0, :Nr, :] = I_components
            X[0, Nr:, :] = Q_components

            theta_cnn = cnn_model.predict(X, verbose=0)[0][0]

            # MUSIC estimation
            _, _, theta_music = music_algorithm(r_noisy, num_sources=1)

            # Calculate errors
            cnn_error = np.abs(theta_cnn - theta_true)
            music_error = (
                np.abs(theta_music[0] - theta_true) if len(theta_music) > 0 else 90
            )

            cnn_rmse += cnn_error**2
            music_rmse += music_error**2

        # Calculate RMSE
        cnn_rmse = np.sqrt(cnn_rmse / num_samples)
        music_rmse = np.sqrt(music_rmse / num_samples)

        results.append((snr, cnn_rmse, music_rmse))
        print(
            f"SNR = {snr} dB: CNN RMSE = {cnn_rmse:.2f}°, MUSIC RMSE = {music_rmse:.2f}°"
        )

    # Plot results
    plt.figure(figsize=(10, 6))
    snrs = [r[0] for r in results]
    cnn_rmses = [r[1] for r in results]
    music_rmses = [r[2] for r in results]

    plt.plot(snrs, cnn_rmses, "o-", label="CNN")
    plt.plot(snrs, music_rmses, "s-", label="MUSIC")
    plt.grid(True)
    plt.xlabel("SNR (dB)")
    plt.ylabel("RMSE (degrees)")
    plt.title("DOA Estimation Performance: CNN vs MUSIC")
    plt.legend()
    plt.show()

    return results


# Main execution code
if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Parameters
    Nr = 8  # Number of antennas
    N_snapshots = 512  # Number of time samples

    # 1. Generate training data for single-source DOA estimation
    print("Generating training data...")
    num_train_samples = 5000
    X, y = generate_data(
        num_train_samples, Nr=Nr, N_snapshots=N_snapshots, snr_range=(-10, 10)
    )

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")

    # 2. Build and train the model
    print("Building and training the model...")
    input_shape = (2 * Nr, N_snapshots)
    model = build_resnet_model(input_shape, output_dim=1, regression=True)

    # Print model summary
    model.summary()

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
        ],
        verbose=1,
    )

    # 3. Evaluate model performance at different SNR levels
    print("Evaluating model performance...")
    snr_values, rmse_results = evaluate_model(model, snr_range=(-20, 20, 5))

    # 4. Compare with MUSIC algorithm
    print("Comparing with MUSIC algorithm...")
    compare_results = compare_cnn_music(model, snr_levels=[-10, 0, 10, 20])

    # 5. Test with more complex scenario similar to your original code
    print("Testing with multi-source scenario...")

    # Generate data with 3 sources
    theta1 = 20
    theta2 = 25
    theta3 = 0

    # Create steering vectors
    s1 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta1 * np.pi / 180)).reshape(
        -1, 1
    )
    s2 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta2 * np.pi / 180)).reshape(
        -1, 1
    )
    s3 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta3 * np.pi / 180)).reshape(
        -1, 1
    )

    # Create tones with different frequencies
    t = np.arange(N) / sample_rate
    tone1 = np.exp(2j * np.pi * 0.01e6 * t).reshape(1, -1)
    tone2 = np.exp(2j * np.pi * 0.02e6 * t).reshape(1, -1)
    tone3 = np.exp(2j * np.pi * 0.03e6 * t).reshape(1, -1)

    # Generate received signal
    r = s1 @ tone1 + s2 @ tone2 + 0.1 * s3 @ tone3

    # Add noise
    n = np.random.randn(Nr, N) + 1j * np.random.randn(Nr, N)
    r_noisy = r + 0.05 * n

    # Apply MUSIC algorithm
    theta_scan, spectrum, doa_estimates = music_algorithm(r_noisy, num_sources=3)

    # Plot MUSIC spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(theta_scan, spectrum)
    plt.grid(True)
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Normalized MUSIC Spectrum (dB)")
    plt.title("MUSIC Spectrum for Multi-Source Scenario")

    # Mark true DOAs
    plt.axvline(theta1, color="r", linestyle="--", label=f"True DOA 1: {theta1}°")
    plt.axvline(theta2, color="g", linestyle="--", label=f"True DOA 2: {theta2}°")
    plt.axvline(theta3, color="b", linestyle="--", label=f"True DOA 3: {theta3}°")

    # Mark estimated DOAs
    for i, doa in enumerate(doa_estimates):
        plt.axvline(doa, color="k", linestyle=":", label=f"Est. DOA {i+1}: {doa:.1f}°")

    plt.legend()
    plt.show()

    print(f"True DOAs: {theta1}°, {theta2}°, {theta3}°")
    print(f"Estimated DOAs using MUSIC: {doa_estimates}")

    print("Done!")
