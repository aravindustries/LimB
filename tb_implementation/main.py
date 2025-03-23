#%%
# LimB: Learning-driven mmWave Beamforming
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm 

csv_file = "../data/aggregated_training_data/agg_data.csv"
df = pd.read_csv(csv_file, header=None)

# Extract angles and gain profiles
angles = df[0].to_numpy(dtype=np.float32)  # First column: angles
gain_profiles = df.iloc[:, 2:65].to_numpy(dtype=np.float32)  # Columns 2 to 64: gain profiles

# Normalize gain profiles
scaler = MinMaxScaler()
gain_profiles_normalized = scaler.fit_transform(gain_profiles)

# Increase training set to 90%
X_train, X_test, y_train, y_test = train_test_split(
    gain_profiles_normalized, angles, test_size=0.05, random_state=42
)

device = torch.device("cuda")
# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # Ensure correct shape
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Create DataLoader with randomized training order
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Randomized order

# Define the MLP model with Dropout
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Output layer for regression
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)  # Dropout with 30% probability

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout during training
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # Linear activation for regression output
        return x

# Initialize model
input_dim = X_train.shape[1]
model = MLP(input_dim)

# Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model with tqdm waitbar
epochs = 1000
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    # Use tqdm for progress bar
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False) as pbar:
        for inputs, targets in pbar:
            optimizer.zero_grad()  # Clear gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, targets)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())  # Update tqdm progress bar

    # Print loss every 10 epochs
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    test_loss = criterion(y_pred, y_test_tensor).item()
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
plt.show()
# %%
