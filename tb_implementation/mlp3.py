import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm 
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import utils2

def get_beams(n, spread=49):
    return np.linspace(32-(spread/2), 32+(spread/2), n, dtype=int)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(input_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.bn3 = nn.BatchNorm1d(1)
        self.fc4 = nn.Linear(input_dim, 1)
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(p=0.9)

    def forward(self, x):
        x1 = self.silu(self.fc1(x))
        x1 = self.bn1(x1)
        x2 = self.fc2(x)
        x2 = self.bn2(x2)
        x3 = x1 * x2
        x3 = self.dropout(x3)
        x4 = self.fc3(x3)
        x4 = self.bn3(x4)
        x5 = self.fc4(x)  # Final output layer
        return x4 + x5


class doaMLP():
    def __init__(self, input_shape):
        self.in_shape = input_shape
        self.model = MLP(input_shape) 
        self.criterion = nn.MSELoss()

    def train_model(self, X_train, y_train, lr, b, epochs=30):
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # Ensure correct shape

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=b, shuffle=True)  # Randomized order

        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

        print(running_loss)
        return running_loss

    def eval_model(self, X_test, y_test):
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_test_tensor)
            test_loss = self.criterion(y_pred, y_test_tensor).item()
            mae = torch.mean(torch.abs(y_pred - y_test_tensor)).item()
 
        print(f"Test Loss (MSE): {test_loss:.4f}")
        print(f"Test MAE: {mae:.4f}")

        # Convert predictions to numpy for visualization
        y_pred_numpy = y_pred.numpy().flatten()
        y_test_numpy = y_test_tensor.numpy().flatten()

        # Plot actual vs predicted angles
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test_numpy, y_pred_numpy, alpha=0.7, color="blue", label="Predictions")
        plt.plot([y_test_numpy.min(), y_test_numpy.max()], [y_test_numpy.min(), y_test_numpy.max()], "r--", label="Ideal Fit")
        plt.xlabel("True Angle (°)")
        plt.ylabel("Predicted Angle (°)")
        plt.title("True vs. Predicted Angles")
        plt.legend()
        plt.grid(True)
        plt.savefig('MLP performance'+str(self.in_shape))
                    
    def iterative_train(self, df_train, scaler, N):    
        beta = get_beams(self.in_shape, 50)
        scaler = MinMaxScaler()
        
        for lr, b in [(0.1, 16), (0.01, 64), (0.001, 256), (0.0001, 1024)]:
            for k in tqdm(range(N), desc="Training Progress"):
                ndf, snr = utils2.adjust_noise_to_target_snr(df_train, np.random.uniform(-5, 20))
                y_train = ndf['Angle'].to_numpy()
                X_train = scaler.fit_transform(ndf.iloc[:, 2:65].iloc[:, beta])
                loss = self.train_model(X_train, y_train, lr=lr, b=b, epochs=30)
# class data_processor():
#    def __init__(self, csv_file):
