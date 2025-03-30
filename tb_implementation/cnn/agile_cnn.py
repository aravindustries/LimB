import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from generate_data import generate_data
from parse_iq import dataIn

class AgileCNN(nn.Module):
    def __init__(self, num_beams, num_classes=65):
        super().__init__()
        self.num_beams = num_beams
        
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def apply_agile_beam_switching(X, y, num_beams, spread=32, Nr=4, d=0.55):
    
    #steering vectors
    betas = np.linspace(-spread, spread, num_beams)
    # need to convert betas into radians dunno if ari remembered that
    beta_matrix = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(betas * np.pi / 180).reshape(-1,1))
    

    k = 32
    alphas = np.array(range(-k, k+1)) * np.pi / 180
    alpha_matrix = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(alphas.reshape(-1, 1)))

    gain = alpha_matrix @ beta_matrix.conj().T / Nr
    # this is power gain
    gain = np.abs(np.square(gain))
    gain /= np.linalg.norm(gain, axis=1).reshape(-1, 1)

    num_samples, _, N_snapshots = X.shape
    segment_size = N_snapshots // num_beams
    X_agile = np.zeros((num_samples, 2, N_snapshots))

    for i in range(num_samples):
        true_angle_idx = int(y[i]) -32
        angle_idx = true_angle_idx + k

        complex_signal = X[i, :Nr] + 1j * X[i, Nr:]


        for b in range(num_beams):
            beam_gain = gain[angle_idx, b]
            start_idx = b * segment_size
            end_idx = (b + 1) * segment_size
            
            beam_weights = beta_matrix[b].conj()
            beamformed_segment = beam_weights @ complex_signal[:, start_idx:end_idx] * beam_gain
            
            X_agile[i, 0, start_idx:end_idx] = np.real(beamformed_segment)
            X_agile[i, 1, start_idx:end_idx] = np.imag(beamformed_segment)
    
    return X_agile

def train_agile_model(
    model,
    train_loader,
    val_loader,
    epochs=30,
    device=torch.device("cpu"),
    learning_rate=0.001,
):
    print(f"Training on device: {device}")
    
    model = model.to(device)
    best_val_acc = 0.0
    best_model_state = None
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )


    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * correct / total

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * correct / total

        scheduler.step(val_acc)

        print(
            f"Epoch {epoch+1}/{epochs}: "
            f"Train Acc: {train_acc:.2f}%, "
            f"Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"New best model: {val_acc:.2f}%")

    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation accuracy: {best_val_acc:.2f}%")
    
    return model


def evaluate_and_plot(model, test_data, test_labels, num_beams, device):
    model.eval()
    model = model.to(device)
    
    X_test = torch.FloatTensor(test_data).to(device)
    y_test = torch.LongTensor(test_labels).to(device)
    
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = outputs.max(1)
    
    predicted_np = predicted.cpu().numpy()
    labels_np = y_test.cpu().numpy()
    
    accuracy = np.mean(predicted_np == labels_np) * 100
    mae = np.mean(np.abs(predicted_np - labels_np))
    
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Mean Absolute Error: {mae:.2f} degrees")
    
    angle_values = np.arange(-32, 33)  # -32 to +32 degrees
    accuracy_by_angle = []
    mae_by_angle = []
    
    for angle_idx, angle in enumerate(angle_values):
        label_value = angle + 32  # Convert from angle to label (0-64)
        mask = (labels_np == label_value)
        
        if np.sum(mask) > 0:
            angle_acc = np.mean(predicted_np[mask] == labels_np[mask]) * 100
            angle_mae = np.mean(np.abs(predicted_np[mask] - labels_np[mask]))
        else:
            angle_acc = 0
            angle_mae = 0
            
        accuracy_by_angle.append(angle_acc)
        mae_by_angle.append(angle_mae)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(angle_values, accuracy_by_angle, 'b-', linewidth=2)
    plt.grid(True)
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Agile CNN with {num_beams} Beams - Accuracy by Angle")
    plt.ylim(0, 105)
    
    plt.subplot(2, 1, 2)
    plt.plot(angle_values, mae_by_angle, 'r-', linewidth=2)
    plt.grid(True)
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Mean Absolute Error (degrees)")
    plt.title(f"Agile CNN with {num_beams} Beams - Error by Angle")
    
    plt.tight_layout()
    plt.savefig(f"agile_results_{num_beams}_beams.png")
    plt.show()
    
    return accuracy, mae, angle_values, accuracy_by_angle, mae_by_angle

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"you are using: {device}")

    # using old generate data
    X, y = generate_data(
        num_samples=10000,
        Nr=4,
        N_snapshots=512,
        snr_range=(-10, 20),
        num_of_thetas_range=(2,3)
    )

    print(X.shape)
    print(y.shape)

    num_beams = 4
    batch_size = 64
    epochs = 40
    learning_rate = 0.001

    X_agile = apply_agile_beam_switching(X, y, num_beams)

    print(X_agile.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X_agile, y, test_size=0.2, random_state=42
    )

    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    print(f"Number of beams: {num_beams}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    model = AgileCNN(num_beams=num_beams, num_classes=65).to(device)

    model = train_agile_model(
        model,
        train_loader,
        test_loader,
        epochs=epochs,
        device=device,
        learning_rate=learning_rate
    )


    accuracy, mae, _, _, _ = evaluate_and_plot(model, X_test, y_test, num_beams, device)
    print(f"Final results: Accuracy: {accuracy:.2f}%, MAE: {mae:.2f} degrees")
   