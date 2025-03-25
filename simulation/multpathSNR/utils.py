import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt

# change this to cpu if you dont have gpu
device = torch.device('cuda')
'''Simulation Parameters for 28 GHZ IBM Phased Array Antenna Module'''
'''DO NOT CHANGE'''

d = 0.45  # distance normalized to wavelength
array_size = 4  
Nr = array_size**2  
sample_rate = 1e6  # sampling rate
N = 1024  # Number of samples

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
    s = steering_vector(torch.tensor(theta_rad, device=device), 
                        torch.tensor(phi_rad, device=device))
    
    X = s @ tx.to(device)
    return X


def create_noisy_tone(theta_rad, phi_rad):
    # s = steering_vector(torch.tensor(theta_rad, device=device), 
    #                     torch.tensor(phi_rad, device=device))
    # X = s @ tx.to(device)
    # return X
    s = steering_vector(torch.tensor(theta_rad, device=device), 
                        torch.tensor(phi_rad, device=device))
    
    # gaussian amplitude noise
    amplitude_noise = torch.randn_like(s, device=device)
    noisy_amplitude = s + amplitude_noise
    # gaussian phase noise
    phase_noise = torch.randn_like(s, device=device)
    noisy_phase = torch.exp(1j * phase_noise) * 0.3 # phase noise must be in exponential form
    # create additive noise
    noisy_signal = noisy_amplitude #* noisy_phase
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

def theta_scan(X, n):
    theta_scan = torch.linspace(-torch.pi/4, torch.pi/4, n, device=device)
    results = torch.zeros(len(theta_scan), 1, device=device)

    for i, theta_i in enumerate(theta_scan):
        w = steering_vector(theta_i, torch.tensor(0))
        X_weighted = w.conj().T @ X
        results[i] = torch.var(X_weighted)
    
    return normalize_minmax(results).cpu().numpy()

# def get_beam_weights(theta_i):
