import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

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


'''Simulation'''

# generate all combinations of theta/phi
theta = torch.linspace(-torch.pi/2, torch.pi/2, 200, device=device)
phi = torch.linspace(-torch.pi/2, torch.pi/2, 200, device=device)

# duplicate them 5 times to account for noisy gain profiles
theta = torch.tile(theta, (5,))
phi = torch.tile(phi, (5,))

# randomize the order of the training angles while 
perm = torch.randperm(len(theta), device=device)

# reorder while saving original values
theta = theta[perm]
phi = phi[perm]

# preallocate lists for variable length simulations
train_X = []
train_y = []

# get gain profiles all angles in training set
for theta_i in tqdm(theta, desc="Processing theta"):
    for phi_j in phi:
        r = getGainProfile(theta_i.item(), phi_j.item())
        train_X.append(r)
        train_y.append([theta_i.item(), phi_j.item()])


'''Save the data'''
import numpy as np
train_X = np.array(train_X)
train_y = np.array(train_y)

print(f'train_X len: {len(train_X)}')
print(f'train_X shape: {train_X[0].shape}')
print(f'train_y len: {len(train_y)}')
print(f'train_y shape: {train_y[0].shape}')

# Save data
np.save('noisy_X.npy', train_X)
np.save('noisy_y.npy', train_y)
