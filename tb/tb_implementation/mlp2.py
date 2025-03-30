import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm 
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pandas as pd
import utils2


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    # elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #     device = torch.device("mps")
    #     print("Using Apple Metal Performance Shaders (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

device = get_device()

def get_beams(n, spread=60):
    # beams = np.linspace(32 - (spread / 2), 32 + (spread / 2), n, dtype=int)
    return np.round(np.linspace(11, 52, n)).astype(int)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(input_dim, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)
        self.bn3 = nn.BatchNorm1d(num_classes)
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(p=0.9)

    def forward(self, x):
        x1 = self.fc1(x)
        x1 = self.bn1(x1)
        x2 = self.fc2(x)
        x2 = self.bn2(x2)
        x3 = self.silu(x1) * x2
        x3 = self.dropout(x3)
        x4 = self.fc3(x3)
        x4 = self.bn3(x4)  # Apply batch normalization to the output layer
        return x4

class doaMLPClassifier():
    def __init__(self, input_shape, num_classes):
        self.in_shape = input_shape
        self.num_classes = num_classes
        self.model = MLPClassifier(input_shape, num_classes).to(device)
        self.criterion = nn.CrossEntropyLoss()

    def train_model(self, X_train, y_train, lr, epochs=30):
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

        print(running_loss)
        return running_loss

    def eval_model(self, X_test, y_test):
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_test_tensor)
            predictions = torch.argmax(logits, dim=1)
        
        return y_test_tensor.cpu().numpy(), predictions.cpu().numpy()


    def iterative_train(self, scaler, N):
        df_train = pd.read_csv('../data_processing/train_gain_prof.csv')
        beta = get_beams(self.in_shape, 60)
        scaler = MinMaxScaler()     
        for lr in [0.001, 0.0005, 0.0001]:
            for k in tqdm(range(N), desc="Training Progress"):
                ndf, snr = utils2.adjust_noise_to_target_snr(df_train, np.random.uniform(0, 10))
                y_train = ndf['Angle'].to_numpy()
                y_train = np.digitize(y_train, bins=np.linspace(-45, 45, self.num_classes)) - 1
                X_train = scaler.fit_transform(ndf.iloc[:, 2:65].iloc[:, beta])

                # y_train = y_train[15:-15]
                # X_train = X_train[15:-15]

                loss = self.train_model(X_train, y_train, lr, epochs=1)
