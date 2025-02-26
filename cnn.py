import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

sample_rate = 1e6
N = 10000
d = 0.5

def generate_data(num_samples, Nr=8, N_snapshots=512, snr_range=(-10, 10)):
    """Generate training data for DOA estimation using IQ components."""
    X = np.zeros((num_samples, 2*Nr, N_snapshots), dtype=np.float32)
    y = np.zeros(num_samples)
    
    for i in range(num_samples):
        theta_degrees = np.random.uniform(-90, 90)
        y[i] = theta_degrees
        
        theta = theta_degrees / 180 * np.pi
        
        s = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta)).reshape(-1, 1)
        
        t = np.arange(N_snapshots) / sample_rate
        f_tone = 0.02e6
        tx = np.exp(2j * np.pi * f_tone * t).reshape(1, -1)
        
        r = s @ tx
        
        snr_db = np.random.uniform(snr_range[0], snr_range[1])
        snr_linear = 10 ** (snr_db / 10)
        
        signal_power = np.mean(np.abs(r) ** 2)
        noise_power = signal_power / snr_linear
        noise_std = np.sqrt(noise_power / 2)  # divide by 2 for complex noise
        
        noise = noise_std * (np.random.randn(Nr, N_snapshots) + 1j * np.random.randn(Nr, N_snapshots))
        
        r = r + noise
        
        I_components = np.real(r)
        Q_components = np.imag(r)
        
        X[i, :Nr, :] = I_components
        X[i, Nr:, :] = Q_components
    
    return X, y

def generate_multi_source_data(num_samples, Nr=2, N_snapshots=512, num_sources=2, min_separation=5, snr_range=(-10, 10)):
    """Generate training data for DOA estimation with multiple sources."""
    X = np.zeros((num_samples, 2*Nr, N_snapshots), dtype=np.float32)
    y = np.zeros((num_samples, num_sources))
    
    for i in range(num_samples):
        # Generate distinct DOA angles with minimum separation
        while True:
            theta_degrees = np.sort(np.random.uniform(-90, 90, num_sources))
            if num_sources == 1 or np.min(np.diff(theta_degrees)) >= min_separation:
                break
        
        y[i, :] = theta_degrees
        
        # Convert to radians
        thetas = theta_degrees / 180 * np.pi
        
        # Initialize received signal
        r = np.zeros((Nr, N_snapshots), dtype=complex)
        
        # Add contribution from each source
        for j, theta in enumerate(thetas):
            # Create steering vector
            s = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta)).reshape(-1, 1)
            
            # Create signal with different frequency for each source
            t = np.arange(N_snapshots) / sample_rate
            f_tone = (0.01 + 0.01 * j) * 1e6  # Different frequency for each source
            tx = np.exp(2j * np.pi * f_tone * t).reshape(1, -1)
            
            # Add to received signal
            r += s @ tx
        
        # Add noise based on SNR
        snr_db = np.random.uniform(snr_range[0], snr_range[1])
        snr_linear = 10 ** (snr_db / 10)
        
        # Calculate signal power
        signal_power = np.mean(np.abs(r) ** 2)
        noise_power = signal_power / snr_linear
        noise_std = np.sqrt(noise_power / 2)
        
        # Generate complex noise
        noise = noise_std * (np.random.randn(Nr, N_snapshots) + 1j * np.random.randn(Nr, N_snapshots))
        
        # Add noise to signal
        r = r + noise
        
        # Extract I and Q components
        I_components = np.real(r)
        Q_components = np.imag(r)
        
        # Stack I and Q components
        X[i, :Nr, :] = I_components
        X[i, Nr:, :] = Q_components
    
    return X, y

# ResNet-based model for DOA estimation (similar to what the paper describes)
def build_resnet_model(input_shape, output_dim=1, regression=True):

    inputs = layers.Input(shape=input_shape)
    
    # Reshape for 2D convolution (adding channel dimension)
    x = layers.Reshape((*input_shape, 1))(inputs)
    
    # First convolutional layer
    x = layers.Conv2D(64, (input_shape[0], 5), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((1, 2))(x)
    
    # Define a residual block
    def residual_block(x, filters, stride=1):
        identity = x
        
        # First convolutional layer in the block
        x = layers.Conv2D(filters, (1, 3), strides=(1, stride), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Second convolutional layer in the block
        x = layers.Conv2D(filters, (1, 3), strides=(1, 1), padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Adjust identity mapping if dimensions change
        if stride > 1 or identity.shape[-1] != filters:
            identity = layers.Conv2D(filters, (1, 1), strides=(1, stride), padding='same')(identity)
            identity = layers.BatchNormalization()(identity)
        
        # Add identity mapping
        x = layers.add([x, identity])
        x = layers.Activation('relu')(x)
        
        return x
    
    # Add residual blocks with increasing channels
    x = residual_block(x, 64, stride=2)
    x = residual_block(x, 64)
    
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)
    
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)
    
    x = residual_block(x, 512, stride=2)
    x = residual_block(x, 512)
    
    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Output layer
    if regression:
        outputs = layers.Dense(output_dim)(x)
    else:
        # For classification, use softmax activation
        num_classes = 181  # -90 to 90 degrees with 1-degree resolution
        outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    if regression:
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='mse',
                     metrics=['mae'])
    else:
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
    
    return model

# Example of training the model
def train_doa_model():
    # Parameters
    Nr = 8
    N_snapshots = 512
    num_samples = 10000
    num_sources = 1  # Single source
    
    # Generate data
    X, y = generate_data(num_samples, Nr=Nr, N_snapshots=N_snapshots, snr_range=(-10, 10))
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build model
    input_shape = (2*Nr, N_snapshots)
    model = build_resnet_model(input_shape, output_dim=1, regression=True)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
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
        X_test, y_test = generate_data(num_test_samples, Nr=Nr, N_snapshots=N_snapshots, 
                                       snr_range=(snr, snr))
        
        # Predict DOA angles
        y_pred = model.predict(X_test)
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((y_pred.flatten() - y_test) ** 2))
        rmse_results.append(rmse)
        
        print(f"SNR = {snr} dB, RMSE = {rmse:.2f} degrees")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(snr_values, rmse_results, 'o-')
    plt.grid(True)
    plt.xlabel('SNR (dB)')
    plt.ylabel('RMSE (degrees)')
    plt.title('DOA Estimation Performance vs SNR')
    plt.show()
    
    return snr_values, rmse_results

def music_algorithm(r, num_sources=1):
    Nr = r.shape[0]
    
    # Calculate covariance matrix
    R = r @ r.conj().T
    
    # Eigenvalue decomposition
    w, v = np.linalg.eig(R)
    
    # Sort eigenvalues and eigenvectors
    eig_val_order = np.argsort(np.abs(w))
    v = v[:, eig_val_order]
    
    # Noise subspace
    V = v[:, :Nr - num_sources]  # Noise subspace is the eigenvectors with smallest eigenvalues
    
    # Scan angles
    theta_scan = np.linspace(-np.pi/2, np.pi/2, 181)  # -90 to 90 degrees
    results = []
    
    for theta_i in theta_scan:
        s = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta_i)).reshape(-1, 1)
        metric = 1 / (s.conj().T @ V @ V.conj().T @ s)  # The main MUSIC equation
        metric = np.abs(metric.squeeze())
        results.append(metric)
    
    results = np.array(results)
    results_db = 10 * np.log10(results / np.max(results))
    
    # Find peaks in the spectrum
    peaks = []
    for i in range(1, len(results_db) - 1):
        if results_db[i] > results_db[i-1] and results_db[i] > results_db[i+1]:
            peaks.append((theta_scan[i] * 180 / np.pi, results_db[i]))
    
    # Sort peaks by amplitude
    peaks.sort(key=lambda x: x[1], reverse=True)
    
    # Return top num_sources peaks as DOA estimates
    doa_estimates = [peak[0] for peak in peaks[:num_sources]]
    
    return theta_scan * 180 / np.pi, results_db, np.array(doa_estimates)

# Function to compare CNN and MUSIC performance
def compare_cnn_music(cnn_model, snr_levels=[-10, 0, 10], num_samples=100):
    Nr = 8
    N_snapshots = 512
    
    results = []
    
    for snr in snr_levels:
        cnn_rmse = 0
        music_rmse = 0
        
        for _ in range(num_samples):
            # Generate a test sample
            theta_true = np.random.uniform(-80, 80)  # Avoid edge cases
            
            # Generate received signal
            theta_rad = theta_true * np.pi / 180
            s = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta_rad)).reshape(-1, 1)
            t = np.arange(N_snapshots) / sample_rate
            f_tone = 0.02e6
            tx = np.exp(2j * np.pi * f_tone * t).reshape(1, -1)
            r = s @ tx
            
            # Add noise
            snr_linear = 10 ** (snr / 10)
            signal_power = np.mean(np.abs(r) ** 2)
            noise_power = signal_power / snr_linear
            noise_std = np.sqrt(noise_power / 2)
            noise = noise_std * (np.random.randn(Nr, N_snapshots) + 1j * np.random.randn(Nr, N_snapshots))
            r_noisy = r + noise
            
            # CNN estimation
            I_components = np.real(r_noisy)
            Q_components = np.imag(r_noisy)
            X = np.zeros((1, 2*Nr, N_snapshots), dtype=np.float32)
            X[0, :Nr, :] = I_components
            X[0, Nr:, :] = Q_components
            
            theta_cnn = cnn_model.predict(X, verbose=0)[0][0]
            
            # MUSIC estimation
            _, _, theta_music = music_algorithm(r_noisy, num_sources=1)
            
            # Calculate errors
            cnn_error = np.abs(theta_cnn - theta_true)
            music_error = np.abs(theta_music[0] - theta_true) if len(theta_music) > 0 else 90
            
            cnn_rmse += cnn_error ** 2
            music_rmse += music_error ** 2
        
        # Calculate RMSE
        cnn_rmse = np.sqrt(cnn_rmse / num_samples)
        music_rmse = np.sqrt(music_rmse / num_samples)
        
        results.append((snr, cnn_rmse, music_rmse))
        print(f"SNR = {snr} dB: CNN RMSE = {cnn_rmse:.2f}°, MUSIC RMSE = {music_rmse:.2f}°")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    snrs = [r[0] for r in results]
    cnn_rmses = [r[1] for r in results]
    music_rmses = [r[2] for r in results]
    
    plt.plot(snrs, cnn_rmses, 'o-', label='CNN')
    plt.plot(snrs, music_rmses, 's-', label='MUSIC')
    plt.grid(True)
    plt.xlabel('SNR (dB)')
    plt.ylabel('RMSE (degrees)')
    plt.title('DOA Estimation Performance: CNN vs MUSIC')
    plt.legend()
    plt.show()
    
    return results

# Main execution code
if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Parameters
    Nr = 8          # Number of antennas
    N_snapshots = 512  # Number of time samples
    
    # 1. Generate training data for single-source DOA estimation
    print("Generating training data...")
    num_train_samples = 5000
    X, y = generate_data(num_train_samples, Nr=Nr, N_snapshots=N_snapshots, snr_range=(-10, 10))
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")
    
    # 2. Build and train the model
    print("Building and training the model...")
    input_shape = (2*Nr, N_snapshots)
    model = build_resnet_model(input_shape, output_dim=1, regression=True)
    
    # Print model summary
    model.summary()
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
        ],
        verbose=1,
    )
    
    # 3. Evaluate model performance at different SNR levels
    print("Evaluating model performance...")
    snr_values, rmse_results = evaluate_model(model, snr_range=(-20, 20, 5))
    
    # 4. Compare with MUSIC algorithm
    print("Comparing with MUSIC algorithm...")
    compare_results = compare_cnn_music(model, snr_levels=[-10, 0, 10, 20])
    
    # 5. Test with more complex scenario similar to your original code
    print("Testing with multi-source scenario...")
    
    # Generate data with 3 sources
    theta1 = 20
    theta2 = 25
    theta3 = 0
    
    # Create steering vectors
    s1 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta1 * np.pi / 180)).reshape(-1, 1)
    s2 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta2 * np.pi / 180)).reshape(-1, 1)
    s3 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta3 * np.pi / 180)).reshape(-1, 1)
    
    # Create tones with different frequencies
    t = np.arange(N) / sample_rate
    tone1 = np.exp(2j * np.pi * 0.01e6 * t).reshape(1, -1)
    tone2 = np.exp(2j * np.pi * 0.02e6 * t).reshape(1, -1)
    tone3 = np.exp(2j * np.pi * 0.03e6 * t).reshape(1, -1)
    
    # Generate received signal
    r = s1 @ tone1 + s2 @ tone2 + 0.1 * s3 @ tone3
    
    # Add noise
    n = np.random.randn(Nr, N) + 1j * np.random.randn(Nr, N)
    r_noisy = r + 0.05 * n
    
    # Apply MUSIC algorithm
    theta_scan, spectrum, doa_estimates = music_algorithm(r_noisy, num_sources=3)
    
    # Plot MUSIC spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(theta_scan, spectrum)
    plt.grid(True)
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Normalized MUSIC Spectrum (dB)')
    plt.title('MUSIC Spectrum for Multi-Source Scenario')
    
    # Mark true DOAs
    plt.axvline(theta1, color='r', linestyle='--', label=f'True DOA 1: {theta1}°')
    plt.axvline(theta2, color='g', linestyle='--', label=f'True DOA 2: {theta2}°')
    plt.axvline(theta3, color='b', linestyle='--', label=f'True DOA 3: {theta3}°')
    
    # Mark estimated DOAs
    for i, doa in enumerate(doa_estimates):
        plt.axvline(doa, color='k', linestyle=':', label=f'Est. DOA {i+1}: {doa:.1f}°')
    
    plt.legend()
    plt.show()
    
    print(f"True DOAs: {theta1}°, {theta2}°, {theta3}°")
    print(f"Estimated DOAs using MUSIC: {doa_estimates}")
    
    print("Done!")
