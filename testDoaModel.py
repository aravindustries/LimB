import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib

# change this to cpu if you dont have gpu
device = torch.device('cuda')
print(f"Using device: {device}")

'''Simulation Parameters for 28 GHZ IBM Phased Array Antenna Module'''
'''DO NOT CHANGE'''

frequency = 28e9  # 28 GHz
c = 3e8
wavelength = c / frequency
element_spacing = 5.9e-3  # 5.9 mm pitch (element spacing)
d = element_spacing / wavelength  # distance normalized to wavelength
array_size = 8  # 8x8 phased array
Nr = array_size**2  # 64 elements total
sample_rate = 1e6  # sampling rate
N = 10000  # Number of samples

''' Simulation Functions'''

# initilize time vector and transmitted signal
t = torch.arange(N, dtype=torch.float32, device=device) / sample_rate
tx = torch.exp(2j * torch.pi * 60e9 * t).reshape(1, -1)

# meshgrid for element positions
x_indices, y_indices = torch.meshgrid(
    torch.arange(array_size, device=device), 
    torch.arange(array_size, device=device), 
    indexing='ij'
)

# center of array is indez zero
x_indices = x_indices.flatten().float() - (array_size - 1) / 2
y_indices = y_indices.flatten().float() - (array_size - 1) / 2


# compute steering vector for transmitted wave originating from a given theta/phi
def steering_vector(theta, phi):
    s = torch.exp(
        1j * 2 * torch.pi * d * (
            x_indices * torch.sin(theta) * torch.cos(phi) +
            y_indices * torch.sin(theta) * torch.sin(phi)
        )
    ).to(device)
    return s.reshape(-1, 1)


def create_tone(theta_rad, phi_rad):
    # s = steering_vector(torch.tensor(theta_rad, device=device), 
    #                     torch.tensor(phi_rad, device=device))
    # X = s @ tx.to(device)
    # return X
    s = steering_vector(torch.tensor(theta_rad, device=device), 
                        torch.tensor(phi_rad, device=device))
    
    # gaussian amplitude noise
    amplitude_noise = torch.randn_like(s, device=device) * 0.3
    noisy_amplitude = s + amplitude_noise
    # gaussian phase noise
    phase_noise = torch.randn_like(s, device=device) * 0.3
    noisy_phase = torch.exp(1j * phase_noise)  # phase noise must be in exponential form
    # create additive noise
    noisy_signal = noisy_amplitude * noisy_phase
    # Add noise to transmitted signal
    X = noisy_signal @ tx.to(device)
    return X

# Function to normalize values
def normalize_minmax(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    return (tensor - min_val) / (max_val - min_val)

# Function to compute gain profile for a given number of beams
def getGainProfile(theta, phi, sqrtNbeams=4):

    # create noisy recieved signal for a given theta/phi
    X = create_tone(theta, phi)

    # Scan angles
    theta_scan = torch.linspace(-torch.pi/3, torch.pi/3, sqrtNbeams, device=device)
    phi_scan = torch.linspace(-torch.pi/3, torch.pi/3, sqrtNbeams, device=device)
    
    results = torch.zeros((len(theta_scan), len(phi_scan)), device=device)
    
    for i, theta_i in enumerate(theta_scan):
        for j, phi_j in enumerate(phi_scan):
            w = steering_vector(theta_i, phi_j) # generate phase weights for each scan angle
            X_weighted = w.conj().T @ X  # apply phase weights to received signal
            results[i, j] = torch.var(torch.abs(X_weighted)) # compute received power
    
    return normalize_minmax(results).cpu().numpy()

'''Simulation to test model'''

def main():
    # test cases with different true theta and phi values
    test_cases = [
        (-34, 46),
        (25, -15),
        (-10, -30),
        (40, 20)
    ]

    fig, axs = plt.subplots(2, 2, figsize=(16, 16))
    axs = axs.ravel()

    # recreate the model architecture
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
                nn.Linear(32, 2)
            )
        
        def forward(self, x):
            return self.network(x)

    # load model
    model = DoAPredictor(16) # flattened gain profile 4x4
    model.load_state_dict(torch.load('best_doa_model.pth'))
    model.eval()

    # Load scaler
    scaler = joblib.load('feature_scaler.joblib')

    for idx, (true_theta_deg, true_phi_deg) in enumerate(test_cases):
        true_theta = np.radians(true_theta_deg)
        true_phi = np.radians(true_phi_deg)

        gain_profile = getGainProfile(true_theta, true_phi)

        gain_profile_flat = gain_profile.reshape(1, -1)
        gain_profile_scaled = scaler.transform(gain_profile_flat)

        with torch.no_grad():
            prediction = model(torch.tensor(gain_profile_scaled, dtype=torch.float32)).numpy()[0]

        pred_theta_deg = np.degrees(prediction[0])
        pred_phi_deg = np.degrees(prediction[1])

        ax = axs[idx]

        im = ax.imshow(gain_profile, extent=(-60, 60, -60, 60), origin='lower', 
               aspect='auto', cmap='viridis')
        plt.colorbar(im, ax=ax, label='Normalized Power')
        
        ax.scatter(true_phi_deg, true_theta_deg, color='red', label='True DoA', marker='o', s=300)
        
        ax.scatter(pred_phi_deg, pred_theta_deg, color='red', label='Predicted DoA', marker='x', s=300)
        
        ax.set_title(f'Case {idx+1}: True DoA (\u03B8={true_theta_deg}, \u03C6={true_phi_deg})')
        ax.set_xlabel('Azimuth Angle \u03C6 (Degrees)')
        ax.set_ylabel('Elevation Angle \u03B8 (Degrees)')
        ax.legend()
        ax.grid(True)

        #summmary
        print(f'Case {idx+1}:')
        print(f'True Direction of Arrival: \u03B8 = {true_theta_deg:.2f} degrees, \u03C6 = {true_phi_deg:.2f} degrees')
        print(f'Predicted Direction of Arrival: \u03B8 = {pred_theta_deg:.2f} degrees, \u03C6 = {pred_phi_deg:.2f} degrees\n')

    plt.tight_layout()
    
    plt.savefig('multiple_doa_gain_profiles.png')
    plt.close()

if __name__ == "__main__":
    main()
