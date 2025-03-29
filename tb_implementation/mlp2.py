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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_beams(n, spread=60):
    return np.linspace(32 - (spread / 2), 32 + (spread / 2), n, dtype=int)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class doaMLPClassifier():
    def __init__(self, input_shape, num_classes):
        self.in_shape = input_shape
        self.num_classes = num_classes
        self.model = MLPClassifier(input_shape, num_classes).to(device)
        self.criterion = nn.CrossEntropyLoss()

    def train_model(self, X_train, y_train, epochs=30):
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

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
        for k in tqdm(range(N), desc="Training Progress"):
            ndf, snr = utils2.adjust_noise_to_target_snr(df_train, np.random.uniform(0, 20))
            y_train = ndf['Angle'].to_numpy()
            y_train = np.clip(y_train, -45, 45)
            y_train = np.digitize(y_train, bins=np.linspace(-45, 45, self.num_classes)) - 1
            X_train = scaler.fit_transform(ndf.iloc[:, 2:65].iloc[:, beta])
            loss = self.train_model(X_train, y_train, epochs=300)
