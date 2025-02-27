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
