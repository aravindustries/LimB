import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

# custom dataset
class DoADataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# MLP Model
class DoAPredictor(nn.Module):
    def __init__(self, input_dim):
        super(DoAPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Predicting theta and phi
        )
    
    def forward(self, x):
        return self.network(x)

# training function
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=3):
    # training
    train_losses = []
    val_losses = []
    
    # best validation loss
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # training phase
        model.train()
        train_epoch_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            # move to device
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            # zero parameter gradients
            optimizer.zero_grad()
            # forward pass
            outputs = model(batch_X)
            # compute loss
            loss = criterion(outputs, batch_y)
            # backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_epoch_loss += loss.item()
        
        # validation
        model.eval()
        val_epoch_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_epoch_loss += loss.item()
        
        # compute average losses
        train_loss = train_epoch_loss / len(train_loader)
        val_loss = val_epoch_loss / len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # summary
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_doa_model.pth')
    
    # # Plot training history
    # plt.figure(figsize=(10,5))
    # plt.plot(train_losses, label='Training Loss')
    # plt.plot(val_losses, label='Validation Loss')
    # plt.title('Model Training History')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('training_history.png')
    # plt.close()

    return train_losses, val_losses

def main():
    device = torch.device('cuda')
    print(f"Using device: {device}")
    
    # load the pregenerated data
    X = np.load('noisy_X.npy')
    y = np.load('noisy_y.npy')
    
    # flatten gain profile before they enter the mlp
    X_flat = X.reshape(X.shape[0], -1)
    
    # train/test split
    X_train, X_val, y_train, y_val = train_test_split(X_flat, y, test_size=0.2, random_state=42)
    
    # scale input features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # save scaler
    joblib.dump(scaler, 'feature_scaler.joblib')
    
    # intialize dataset and dataloaders
    train_dataset = DoADataset(X_train_scaled, y_train)
    val_dataset = DoADataset(X_val_scaled, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # initialize model
    input_dim = X_train_scaled.shape[1]
    model = DoAPredictor(input_dim).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Train and save the model!!
    train_losses, val_losses = train_model(model, train_loader, val_loader, 
                                           criterion, optimizer, device)
    

if __name__ == "__main__":
    main()
    