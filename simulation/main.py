from types import MethodType

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import trange

from agile import Agile
from archive.cnn import *
from generate_data import generate_data
from music import Music


def evaluate_model(model, X_test, y_test, device=None, verbose=False):
    if device is None:
        device = next(model.parameters()).device

    # Move model to device if not already there
    model.to(device)

    # Predict DOA angle classes  # Wait why am I converting to tensor here ? Shouldn't it be done in the data generation ?
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

    if verbose:
        print(f"Accuracy = {accuracy.item() * 100:.2f}%")
        print(f"Mean absolute angle error = {mean_angle_error.item():.2f} degrees")

    return accuracy.item(), mean_angle_error.item()


def make_the_nice_plots(models, names, device=None, snr_levels=[-10, 0, 10]):
    snr = 100
    Nr = 16
    N_snapshots = 512 

    accuracies, maes = [np.zeros(65) for _ in models], [np.zeros(65) for _ in models]

    # This i going one by one through each sample
    for theta in trange(-32, 33):  # also undefined
        X, y = generate_data(
            200,
            Nr=Nr,
            N_snapshots=N_snapshots,
            snr_range=(snr, snr),
            theta_range=(theta, theta),
        )

        for i, model in enumerate(models):
            accuracy, mae = model.evaluate(X, y)
            accuracies[i][theta+32] = accuracy * 100
            maes[i][theta+32] = mae

    plt.figure(figsize=(12, 10))

    plt.subplot(2, 1, 1)
    for i, name in enumerate(names):
        plt.plot(np.arange(-32, 33), accuracies[i], label=name)
    plt.grid(True)
    plt.xlabel("True Angle (degrees)")
    plt.ylabel("Accuracy (%)")
    plt.title(f"DOA Estimation Accuracy: CNN vs MUSIC (SNR = {snr} dB)")
    plt.legend()
    plt.ylim(0, 105)

    plt.subplot(2, 1, 2)
    for i, name in enumerate(names):
        plt.plot(np.arange(-32, 33), maes[i], label=name)
    plt.grid(True)
    plt.xlabel("True Angle (degrees)")
    plt.ylabel("Mean Absolute Error (degrees)")
    plt.title(f"DOA Estimation Error: CNN vs MUSIC (SNR = {snr} dB)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("./plot.png")


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
    Nr = 16  # Actual number from Sivers
    d = 0.55  # I think
    N_snapshots = 512  # Number of time samples

    music = Music(d)
    agile8 = Agile(3, 60)
    agile16 = Agile(16, 60)

    # print("Generating training data...")
    # num_train_samples = 10000
    # X, y = generate_data(
    #     num_train_samples, Nr=Nr, N_snapshots=N_snapshots, snr_range=(-20, 10)
    # )

    # # Split data into training and validation sets  # But we're not really using it ?
    # X_train, X_val, y_train, y_val = train_test_split(
    #     X, y, test_size=0.2, random_state=42
    # )
    # print(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")

    # # 2. Build and train the model
    # print("Building and training the model...")
    # input_shape = (2 * Nr, N_snapshots)

    # Pass the device object, not the function
    # model = train_model(input_shape, X_train, y_train, X_val, y_val, device=device)
    # model.evaluate = MethodType(evaluate_model, model)

    # 3. Evaluate model performance
    print("Evaluating model performance...")
    # evaluate_model(model, X_val, y_val, device=device)
    # make_the_nice_plots([model, music], device)
    # make_the_nice_plots([model, music], ["CNN", "MUSIC"], device)
    make_the_nice_plots(
        [agile8, agile16],
        ["Agile (B=8)", "Agile (B=16)"],
    )

    # 4. Compare with MUSIC algorithm
    # compare_results = compare_cnn_music(
    #     model, snr_levels=[-20, -10, 0, 10], device=device
    # )

    print("Done!")
