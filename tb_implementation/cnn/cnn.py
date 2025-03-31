import torch
import torch.nn as nn
from torch.optim import Adam


class CNNModel(nn.Module):
    def __init__(self, nums_beams, hidden_dim=128):
        super().__init__()

        self.num_beams = nums_beams

        self.conv1 = nn.Conv1d(2, 4*hidden_dim, 5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(4*hidden_dim)
        self.conv2 = nn.Conv1d(4*hidden_dim, 2*hidden_dim, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(2*hidden_dim)
        self.conv3 = nn.Conv1d(2*hidden_dim, hidden_dim, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.8)
    

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.gelu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.gelu(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = x.squeeze(2)
        x = self.fc(x)

        return x


class CNN:
    def __init__(self, B):
        self.model = CNNModel(B)
    
    def train(self, train_loader, val_loader, lr=0.01, weight_decay=1e-4, epochs=30, patience=5):
    
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        
        self.model = self.model.to(device)
        
        criterion = nn.MSELoss()
        optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Learning rate scheduler now using training loss instead of validation loss
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode='min', factor=0.5, patience=3, verbose=True
        # )
        
        # Variables for early stopping
        best_val_loss = float('inf')
        best_model_state = None
        early_stop_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device).float()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets.unsqueeze(1))
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Update learning rate based on training loss
            # scheduler.step(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device).float()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets.unsqueeze(1))
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            # Print statistics
            print(f'Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
            
        #     # Early stopping logic (still using validation loss to decide if model improved)
        #     # If you want to change this too, replace avg_val_loss with avg_train_loss below
        #     if avg_val_loss < best_val_loss:
        #         best_val_loss = avg_val_loss
        #         best_model_state = self.model.state_dict().copy()
        #         early_stop_counter = 0
        #         print(f"New best model with validation loss: {best_val_loss:.6f}")
        #     else:
        #         early_stop_counter += 1
        #         if early_stop_counter >= patience:
        #             print(f"Early stopping triggered after {epoch+1} epochs")
        #             break
        
        # # Load the best model
        # if best_model_state:
        #     self.model.load_state_dict(best_model_state)
        #     print(f"Loaded best model with validation loss: {best_val_loss:.6f}")
        
        return self.model