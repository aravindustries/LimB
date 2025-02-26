import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

from generate_data import generate_data


# ResNet-based model for DOA estimation (similar to what the paper describes)
def build_resnet_model(input_shape, output_dim=1, regression=True):

    inputs = layers.Input(shape=input_shape)

    # Reshape for 2D convolution (adding channel dimension)
    x = layers.Reshape((*input_shape, 1))(inputs)

    # First convolutional layer
    x = layers.Conv2D(64, (input_shape[0], 5), strides=(1, 1), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((1, 2))(x)

    # Define a residual block
    def residual_block(x, filters, stride=1):
        identity = x

        # First convolutional layer in the block
        x = layers.Conv2D(filters, (1, 3), strides=(1, stride), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        # Second convolutional layer in the block
        x = layers.Conv2D(filters, (1, 3), strides=(1, 1), padding="same")(x)
        x = layers.BatchNormalization()(x)

        # Adjust identity mapping if dimensions change
        if stride > 1 or identity.shape[-1] != filters:
            identity = layers.Conv2D(
                filters, (1, 1), strides=(1, stride), padding="same"
            )(identity)
            identity = layers.BatchNormalization()(identity)

        # Add identity mapping
        x = layers.add([x, identity])
        x = layers.Activation("relu")(x)

        return x

    # Add residual blocks with increasing channels
    x = residual_block(x, 64, stride=2)
    # x = residual_block(x, 64)

    x = residual_block(x, 128, stride=2)
    # x = residual_block(x, 128)

    x = residual_block(x, 256, stride=2)
    # x = residual_block(x, 256)

    x = residual_block(x, 512, stride=2)
    # x = residual_block(x, 512)

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Output layer
    if regression:
        outputs = layers.Dense(output_dim)(x)
    else:
        # For classification, use softmax activation
        num_classes = 181  # -90 to 90 degrees with 1-degree resolution
        outputs = layers.Dense(num_classes, activation="softmax")(x)

    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)

    # Compile model
    if regression:
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    else:
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    return model


# Example of training the model
def train_doa_model():
    # Parameters
    Nr = 8
    N_snapshots = 512
    num_samples = 10000
    num_sources = 1  # Single source

    # Generate data
    X, y = generate_data(
        num_samples, Nr=Nr, N_snapshots=N_snapshots, snr_range=(-10, 10)
    )

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build model
    input_shape = (2 * Nr, N_snapshots)
    model = build_resnet_model(input_shape, output_dim=1, regression=True)

    # Train model
    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
        ],
    )

    return model, history


def evaluate_model(model, snr_range=(-25, 25, 5)):
    Nr = 8
    N_snapshots = 512
    num_test_samples = 1000

    rmse_results = []
    snr_values = range(snr_range[0], snr_range[1] + 1, snr_range[2])

    for snr in snr_values:
        # Generate test data at specific SNR
        X_test, y_test = generate_data(
            num_test_samples, Nr=Nr, N_snapshots=N_snapshots, snr_range=(snr, snr)
        )

        # Predict DOA angles
        y_pred = model.predict(X_test)

        # Calculate RMSE
        rmse = np.sqrt(np.mean((y_pred.flatten() - y_test) ** 2))
        rmse_results.append(rmse)

        print(f"SNR = {snr} dB, RMSE = {rmse:.2f} degrees")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(snr_values, rmse_results, "o-")
    plt.grid(True)
    plt.xlabel("SNR (dB)")
    plt.ylabel("RMSE (degrees)")
    plt.title("DOA Estimation Performance vs SNR")
    plt.show()

    return snr_values, rmse_results
