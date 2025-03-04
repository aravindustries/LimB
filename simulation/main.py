import matplotlib.pyplot as plt
import numpy as np
# import tensorflow as tf
import torch
from sklearn.model_selection import train_test_split

from cnn import *
from generate_data import generate_data
from music import music_algorithm

sample_rate = 1e6
N = 1e4
d = 0.5


def evaluate_model(model, X_test, y_test, device=None):
    if device is None:
        device = next(model.parameters()).device

    # Move model to device if not already there
    model.to(device)

    # Predict DOA angle classes
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        y_pred_classes = torch.argmax(y_pred, axis=1)

        # Calculate accuracy
        results = y_pred_classes == y_test_tensor
        accuracy = torch.mean(results.float())

        # Calculate angle errors
        angle_errors = torch.abs(y_pred_classes - y_test_tensor)
        mean_angle_error = torch.mean(angle_errors.float())

    print(f"Accuracy = {accuracy.item() * 100:.2f}%")
    print(f"Mean absolute angle error = {mean_angle_error.item():.2f} degrees")

    return accuracy.item(), mean_angle_error.item()


def compare_cnn_music(cnn_model, snr_levels=[-10, 0, 10], num_samples=100, device=None):
    if device is None:
        device = next(cnn_model.parameters()).device

    # Move model to device
    cnn_model.to(device)
    cnn_model.eval()

    Nr = 8
    N_snapshots = 512
    results = []

    # Define accuracy thresholds for evaluation
    thresholds = [1, 3, 5]  # Within 1°, 3°, and 5° considered accurate

    for snr in snr_levels:
        cnn_correct = {t: 0 for t in thresholds}
        music_correct = {t: 0 for t in thresholds}

        for i in range(num_samples):
            # Generate a test sample
            theta_true = np.random.uniform(-32, 32)  # Avoid edge cases
            theta_true_class = int(theta_true + 32)  # Convert to class index

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

            # Move tensor to appropriate device
            X_tensor = torch.FloatTensor(X).to(device)

            with torch.no_grad():
                y_pred = cnn_model(X_tensor)[0]
                theta_cnn_class = torch.argmax(y_pred)
                theta_cnn = theta_cnn_class.cpu().numpy() - 32  # Convert back to angle

            # MUSIC estimation
            _, _, theta_music = music_algorithm(r_noisy, num_sources=1)

            if len(theta_music) > 0:
                theta_music_val = theta_music[0]
            else:
                theta_music_val = 0  # Default if MUSIC fails to find a peak

            # Calculate errors
            cnn_error = np.abs(theta_cnn - theta_true)
            music_error = np.abs(theta_music_val - theta_true)

            # Check accuracy within thresholds
            for t in thresholds:
                if cnn_error <= t:
                    cnn_correct[t] += 1
                if music_error <= t:
                    music_correct[t] += 1

        # Calculate accuracy percentages
        cnn_accuracies = {
            t: (count / num_samples) * 100 for t, count in cnn_correct.items()
        }
        music_accuracies = {
            t: (count / num_samples) * 100 for t, count in music_correct.items()
        }

        results.append((snr, cnn_accuracies, music_accuracies))

        # Print results
        print(f"SNR = {snr} dB:")
        for t in thresholds:
            print(
                f"  Within {t}°: CNN = {cnn_accuracies[t]:.1f}%, MUSIC = {music_accuracies[t]:.1f}%"
            )

    # Plot results for the first threshold
    plt.figure(figsize=(10, 6))
    snrs = [r[0] for r in results]

    for t in thresholds:
        cnn_acc = [r[1][t] for r in results]
        music_acc = [r[2][t] for r in results]

        plt.plot(snrs, cnn_acc, "o-", label=f"CNN (±{t}°)")
        plt.plot(snrs, music_acc, "s--", label=f"MUSIC (±{t}°)")

    plt.grid(True)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy (%)")
    plt.title("DOA Estimation Accuracy: CNN vs MUSIC")
    plt.legend()
    plt.ylim(0, 100)
    plt.show()

    return results


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Metal Performance Shaders (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


if __name__ == "__main__":
    np.random.seed(42)

    device = get_device()

    # Parameters
    Nr = 8  # Number of antennas
    N_snapshots = 512  # Number of time samples

    print("Generating training data...")
    num_train_samples = 10000
    X, y = generate_data(
        num_train_samples, Nr=Nr, N_snapshots=N_snapshots, snr_range=(-20, 10)
    )

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")

    # 2. Build and train the model
    print("Building and training the model...")
    input_shape = (2 * Nr, N_snapshots)

    # Pass the device object, not the function
    model = train_model(input_shape, X_train, y_train, X_val, y_val, 64, device=device)

    # 3. Evaluate model performance
    print("Evaluating model performance...")
    evaluate_model(model, X_val, y_val, device=device)

    # 4. Compare with MUSIC algorithm
    compare_results = compare_cnn_music(
        model, snr_levels=[-20, -10, 0, 10], device=device
    )

    print("Done!")
