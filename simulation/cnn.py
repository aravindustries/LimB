import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

from generate_data import generate_data


# ResNet-based model for DOA estimation (similar to what the paper describes)
def build_resnet_model(input_shape):

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

    # For classification, use softmax activation
    num_classes = 65  # -32 to 32 degrees with 1-degree resolution
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def evaluate_model(model, snr_range=(-25, 25, 5)):
    """Evaluate model performance at different SNR levels for classification."""
    Nr = 8
    N_snapshots = 512
    num_test_samples = 1000

    accuracy_results = []
    snr_values = range(snr_range[0], snr_range[1] + 1, snr_range[2])

    for snr in snr_values:
        # Generate test data at specific SNR
        X_test, y_test = generate_data(
            num_test_samples, Nr=Nr, N_snapshots=N_snapshots, snr_range=(snr, snr)
        )

        y_test_classes = (y_test + 32).astype(int)

        # Predict DOA angle classes
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Calculate accuracy
        accuracy = np.mean(y_pred_classes == y_test_classes)
        accuracy_results.append(accuracy * 100)  # Convert to percentage

        print(f"SNR = {snr} dB, Accuracy = {accuracy * 100:.2f}%")

        # You might also want to measure how close the predictions are
        angle_errors = np.abs(y_pred_classes - y_test_classes)
        mean_angle_error = np.mean(angle_errors)
        print(f"Mean absolute angle error = {mean_angle_error:.2f} degrees")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(snr_values, accuracy_results, "o-")
    plt.grid(True)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy (%)")
    plt.title("DOA Estimation Classification Accuracy vs SNR")
    plt.ylim(0, 100)  # Set y-axis from 0 to 100%
    plt.show()

    return snr_values, accuracy_results
