import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm 
import pandas as pd

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
class doaMLP():
    def __init__(self, input_shape):
        self.in_shape = input_shape
        self.model = MLP(input_shape) 
        self.criterion = nn.MSELoss()

    def train_model(
        self,
        X_train,
        y_train,
        epochs=30,
    ):  
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # Ensure correct shape

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Randomized order
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False) as pbar:
                for inputs, targets in pbar:
                    optimizer.zero_grad()  
                    outputs = self.model(inputs)  
                    loss = self.criterion(outputs, targets) 
                    loss.backward()  
                    optimizer.step() 
                    running_loss += loss.item()
                    pbar.set_postfix(loss=loss.item())  

            if (epoch+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

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
        plt.savefig('MLP performance16')


class data_processor():
    def __init__(self, csv_file):
        pa
