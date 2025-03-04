import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from generate_data import generate_data

# Assuming generate_data is available as in the original code
# from generate_data import generate_data


# ResNet-based model for DOA estimation
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # First convolutional layer in the block
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, 3),
            stride=(1, stride),
            padding=(0, 1),
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Second convolutional layer in the block
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(1, 3),
            stride=(1, 1),
            padding=(0, 1),
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride > 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(1, 1),
                    stride=(1, stride),
                    padding=(0, 0),
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = self.relu(out)

        return out


class ResNetDOA(nn.Module):
    def __init__(self, input_shape, num_classes=65):
        super(ResNetDOA, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(
            1, 64, kernel_size=(input_shape[0], 5), stride=(1, 1), padding=(0, 2)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        # Residual blocks
        self.layer1 = ResidualBlock(64, 64, stride=2)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)
        self.layer4 = ResidualBlock(256, 512, stride=2)

        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer
        self.fc = nn.Linear(512, num_classes)

        # For reshaping input
        self.input_shape = input_shape

    def forward(self, x):
        # Reshape for 2D convolution (adding channel dimension)
        batch_size = x.size(0)
        x = x.view(batch_size, 1, self.input_shape[0], self.input_shape[1])

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# Function to initialize and return the model (similar to build_resnet_model in the original)
def build_resnet_model_pytorch(input_shape, num_classes=65):
    model = ResNetDOA(input_shape, num_classes)
    return model


# Example of model usage:
def train_model(
    input_shape,
    X_train,
    y_train,
    X_val,
    y_val,
    batch_size=32,
    epochs=30,
    checkpoint_dir="checkpoints",
    force_retrain=False,
    min_val_acc=85.0,
    dynamic_data=True,
    device=torch.device("cpu"),
):
    import os

    from generate_data import generate_data

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "cnn.pth")

    # Initialize best_val_acc variable
    best_val_acc = 0.0

    # Initialize model
    model = build_resnet_model_pytorch(input_shape)

    if os.path.exists(checkpoint_path) and not force_retrain:
        print(f"Found existing checkpoint at {checkpoint_path}")
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Evaluate the loaded model to see if we should keep it
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        # Convert validation data to tensors
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Evaluate on validation set
        criterion = nn.CrossEntropyLoss()
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
        print(f"Loaded model validation accuracy: {val_acc:.2f}%")

        # If validation accuracy is good enough, return the loaded model
        if val_acc >= min_val_acc:
            print(
                f"Loaded model meets accuracy threshold ({val_acc:.2f}% >= {min_val_acc}%), using it."
            )
            return model
        else:
            print(
                f"Loaded model below accuracy threshold ({val_acc:.2f}% < {min_val_acc}%), retraining."
            )
    else:
        if force_retrain:
            print("Force retrain flag is set, training from scratch.")
        else:
            print(f"No checkpoint found at {checkpoint_path}, training from scratch.")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Calculate the size of the original training set
    num_train_samples = X_train.shape[0]

    # Validation data to tensors (this doesn't change)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    for epoch in range(epochs):
        # Generate new training data for each epoch if dynamic_data is True
        if dynamic_data and epoch > 0:  # Skip first epoch to have a baseline
            print(f"Generating new training data for epoch {epoch+1}...")
            Nr = X_train.shape[1] // 2  # Extract Nr from the data shape
            N_snapshots = X_train.shape[2]

            # Generate new training data with the same shape as the original
            X_train_new, y_train_new = generate_data(
                num_train_samples, Nr=Nr, N_snapshots=N_snapshots, snr_range=(-20, 10)
            )

            # Convert new data to tensors
            X_train_tensor = torch.FloatTensor(X_train_new)
            y_train_tensor = torch.LongTensor(y_train_new)
        else:
            # Use original data for the first epoch
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.LongTensor(y_train)

        # Create training dataset and loader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        # Validation phase
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

        val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * correct / total

        print(
            f"Epoch {epoch+1}/{epochs}: "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        # Save checkpoint if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"Saving checkpoint with validation accuracy: {val_acc:.2f}%")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )

        # Early stopping check
        if epoch >= 14 and not dynamic_data:
            print(
                "Reached epoch 14, stopping to prevent overfitting (set dynamic_data=True to continue with fresh data)"
            )
            break

    # Load the best model (if we improved over initial)
    if best_val_acc > 0:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"Loaded best model with validation accuracy: {checkpoint['val_acc']:.2f}%"
        )

    return model


if __name__ == "__main__":
    # Example usage
    # Assuming generate_data is available
    X, y = generate_data(1000)
    input_shape = X.shape[1:]  # Remove batch dimension
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = train_model(input_shape, X_train, y_train, X_val, y_val, epochs=10)


# Example usage:
"""
# Assuming generate_data is available
X, y = generate_data()
input_shape = X.shape[1:]  # Remove batch dimension
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model = train_model(input_shape, X_train, y_train, X_val, y_val)
"""
