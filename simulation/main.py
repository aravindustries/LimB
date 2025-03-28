from types import MethodType

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange

from agile import Agile
# from archive.cnn import *
from generate_data import generate_data
from music import Music


def evaluate_model(model, X_test, y_test, Nr=16, device=None, verbose=False):
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


def same_data_different_models(models, names, Nr=16, snr=5):
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
            num_of_thetas=(1,2)
        )

        for i, model in enumerate(models):
            accuracy, mae = model.evaluate(X, y)
            accuracies[i][theta+32] = accuracy * 100
            maes[i][theta+32] = mae

    # plt.figure(figsize=(12, 10))

    # plt.subplot(2, 1, 1)
    # for i, name in enumerate(names):
    #     plt.plot(np.arange(-32, 33), accuracies[i], label=name)
    # plt.grid(True)
    # plt.xlabel("True AoA α (°)")
    # plt.ylabel("Accuracy (%)")
    # plt.title(f"DOA Estimation Accuracy: CNN vs MUSIC (SNR = {snr} dB)")
    # plt.legend()
    # plt.ylim(0, 105)

    # plt.subplot(2, 1, 2)
    for i, name in enumerate(names):
        plt.plot(np.arange(-32, 33), maes[i], label=name)
    plt.grid(True)
    plt.xlabel("True AoA α (°)")
    plt.ylabel("Mean angle error (°)")
    plt.title(f"DOA Estimation Error (SNR = {snr} dB)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("./plot.png")

    for i, name in enumerate(names):
        print(f"{name} - Accuracy: {accuracies[i].mean():.2f}%, MAE: {maes[i].mean():.2f} degrees")


def different_data_same_model(model, name, snr_levels=[0, 5, 10, 15, 20], Nr=16):
    accuracies, maes = [np.zeros(65) for _ in snr_levels], [np.zeros(65) for _ in snr_levels]

    for i, snr in enumerate(tqdm(snr_levels)):
        for theta in range(-32, 33):
            X, y = generate_data(
                200,
                Nr=Nr,
                N_snapshots=512,
                snr_range=(snr, snr),
                theta_range=(theta, theta),
                num_of_thetas=(2, 3)
            )

            accuracy, mae = model.evaluate(X, y)
            accuracies[i][theta + 32] = accuracy * 100
            maes[i][theta + 32] = mae
    
    for i, snr in enumerate(snr_levels):
        plt.plot(np.arange(-32, 33), maes[i], label=f"SNR = {snr} dB")
    plt.grid(True)
    plt.xlabel("True AoA α (°)")
    plt.ylabel("Mean angle error (°)")
    plt.title(f"DOA Estimation Error: {name}")
    plt.legend()

    plt.tight_layout()
    plt.savefig("./plot2.png")

    for i, snr in enumerate(snr_levels):
        print(f"SNR = {snr} dB - Accuracy: {accuracies[i].mean():.2f}%, MAE: {maes[i].mean():.2f} degrees")


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

    Nr = 16
    d = 0.55
    N_snapshots = 512

    agile3 = Agile(3, 30, Nr=Nr)
    agile4 = Agile(4, 30, Nr=Nr)
    agile6 = Agile(6, 30, Nr=Nr)
    agile8 = Agile(8, 30, Nr=Nr)
    agile12 = Agile(12, 30, Nr=Nr)
    agile16 = Agile(16, 30, Nr=Nr)

    # same_data_different_models(
    #     [agile3, agile4, agile6, agile8, agile12, agile16],
    #     ["Agile (B=3)", "Agile (B=4)", "Agile (B=6)", "Agile (B=8)", "Agile (B=12)", "Agile (B=16)"],
    #     Nr=Nr,
    # )

    different_data_same_model(
        agile8,
        "Agile (B=8)",
        Nr=Nr,
    )

    print("Done!")
