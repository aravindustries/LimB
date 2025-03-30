import numpy as np
import torch
from parse_iq import dataIn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from iq_psd import *


def normalize_data(X):

    X_norm = np.zeros_like(X)

    for i in range(X.shape[1]):
        mean = np.mean(X[:, i, :])
        std = np.std(X[:, i, :])

        if std > 0:
            X_norm[:, i, :] = (X[:, i, :] - mean) / std
        else:
            X_norm[:, i, :] = X[:, i, :]

    return X_norm


def prepare_data_for_cnn(file_path="Combined_Beam_IQ.csv", num_beams=4, test=True):

    if test:
        data = dataIn(file_path)
    else:
        data = get_aug_data(-10)

    num_angles, total_beams, signal_length = data.shape
    print(f"Parsed data shape: {data.shape}")

    original_angles = np.linspace(-45, 45, num_angles)

    valid_indices = np.where((original_angles >= -45) & (original_angles <= 45))[0]
    clipped_angles = original_angles[valid_indices]

    rounded_angles = np.round(clipped_angles).astype(int)

    labels = rounded_angles + 45

    clipped_data = data[valid_indices]

    selected_beams = np.linspace(0, total_beams - 1, num_beams, dtype=int)
    print(f"Selected beams: {selected_beams}")
    print(f"Valid angles range: {clipped_angles.min()} to {clipped_angles.max()}")
    print(f"Number of valid angles: {len(clipped_angles)}")

    samples_per_beam = 512 // num_beams

    X = np.zeros(
        (len(valid_indices), 2, 512)
    )  # 2 channels for real and imaginary parts
    y = labels

    for i in range(len(valid_indices)):
        current_position = 0
        for b_idx, beam in enumerate(selected_beams):
            complex_signal = clipped_data[i, beam, :]

            if len(complex_signal) > samples_per_beam:
                step = len(complex_signal) // samples_per_beam
                complex_signal = complex_signal[::step][:samples_per_beam]

            actual_samples = min(samples_per_beam, len(complex_signal))

            X[i, 0, current_position : current_position + actual_samples] = np.real(
                complex_signal[:actual_samples]
            )
            X[i, 1, current_position : current_position + actual_samples] = np.imag(
                complex_signal[:actual_samples]
            )

            current_position += actual_samples

    print(f"Processed data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Unique labels: {np.unique(y)}")

    X = normalize_data(X)

    return X, y


# def prepare_datasets(X, y, batch_size=64, test_size=0.2):
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=test_size, random_state=42
#     )

#     print(f"Training samples: {len(y_train)}")
#     print(f"Testing samples: {len(y_test)}")

#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#     elif torch.backends.mps.is_available():
#         device = torch.device("mps")
#     else:
#         device = torch.device("cpu")

#     X_train_tensor = torch.FloatTensor(X_train).to(device)
#     y_train_tensor = torch.LongTensor(y_train).to(device)
#     X_test_tensor = torch.FloatTensor(X_test).to(device)
#     y_test_tensor = torch.LongTensor(y_test).to(device)

#     train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
#     test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

#     train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=min(batch_size, len(test_dataset)))

#     return train_loader, test_loader, X_train, y_train, X_test, y_test, device

if __name__ == "__main__":
    X, y = prepare_data_for_cnn(
        file_path="Combined_Beam_IQ.csv", num_beams=4, test=False
    )
    X_test, y_test = prepare_data_for_cnn(
        file_path="Combined_Beam_IQ.csv", num_beams=4, test=True
    )

    print(f"Total samples available: {len(y)}")
    print(f"Unique angle labels: {np.unique(y)}")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.LongTensor(y).to(device)

    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)

    dataset = TensorDataset(X_tensor, y_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    batch_size = min(32, len(dataset))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size)

    from agile_cnn import AgileCNN, train_agile_model, evaluate_and_plot

    model = AgileCNN(num_beams=4, num_classes=91).to(device)

    model = train_agile_model(
        model,
        data_loader,
        test_data_loader,
        epochs=200,
        device=device,
        learning_rate=0.001,
    )

    accuracy, mae, _, _, _ = evaluate_and_plot(model, X_test, y_test, 4, device)
    print(f"Accuracy on all data: {accuracy:.2f}%, MAE: {mae:.2f} degrees")
