import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.animation import FuncAnimation

sample_rate = 1e6
N = 10000 # number of samples to simulate

# Create a tone to act as the transmitted signal
t = np.arange(N)/sample_rate
f_tone = 0.02e6
tx = np.exp(2j*np.pi*f_tone*t)

# Simulate three omnidirectional antennas in a line with 1/2 wavelength between adjancent ones, receiving a signal that arrives at an angle

d = 0.5
Nr = 3
theta_degrees = 20 # direction of arrival
theta = theta_degrees / 180 * np.pi # convert to radians
s = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta)) # steering vector
#print(s)

# we have to do a matrix multiplication of s and tx, which currently are both 1D, so we have to make them 2D with reshape
s = s.reshape(-1,1)
#print(s.shape) # 3x1
tx = tx.reshape(1,-1) # make a row vector
print(tx.shape) # 1x10000

# so how do we use this? simple:
r = s @ tx # matrix multiply
#print(r.shape) # 3x10000.  r is now going to be a 2D array, 1d is time and 1d is spatial

# more complex scenario
Nr = 8
theta1 = 20 / 180 * np.pi # convert to radians
theta2 = 25 / 180 * np.pi
theta3 = -0 / 180 * np.pi
s1 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta1)).reshape(-1,1) # 8x1
s2 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta2)).reshape(-1,1)
s3 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta3)).reshape(-1,1)
# we'll use 3 different frequencies.  1xN
tone1 = np.exp(2j*np.pi*0.01e6*t).reshape(1,-1)
tone2 = np.exp(2j*np.pi*0.02e6*t).reshape(1,-1)
tone3 = np.exp(2j*np.pi*0.03e6*t).reshape(1,-1)
r = s1 @ tone1 + s2 @ tone2 + 0.1 * s3 @ tone3
n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
r = r + 0.05*n # 8xN

# MUSIC Algorithm (part that doesn't change with theta_i)
num_expected_signals = 3 # Try changing this!
R = r @ r.conj().T # Calc covariance matrix, it's Nr x Nr
w, v = np.linalg.eig(R) # eigenvalue decomposition, v[:,i] is the eigenvector corresponding to the eigenvalue w[i]

if False:
    fig, (ax1) = plt.subplots(1, 1, figsize=(7, 3))
    ax1.plot(10*np.log10(np.abs(w)),'.-')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Eigenvalue [dB]')
    plt.show()
    #fig.savefig('../_images/doa_eigenvalues.svg', bbox_inches='tight') # I EDITED THIS ONE IN INKSCAPE!!!
    exit()

eig_val_order = np.argsort(np.abs(w)) # find order of magnitude of eigenvalues
v = v[:, eig_val_order] # sort eigenvectors using this order
V = np.zeros((Nr, Nr - num_expected_signals), dtype=np.complex64) # Noise subspace is the rest of the eigenvalues
for i in range(Nr - num_expected_signals):
    V[:, i] = v[:, i]

theta_scan = np.linspace(-1*np.pi, np.pi, 1000) # 100 different thetas between -180 and +180 degrees
results = []
for theta_i in theta_scan:
    s = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta_i)).reshape(-1,1)
    metric = 1 / (s.conj().T @ V @ V.conj().T @ s) # The main MUSIC equation
    metric = np.abs(metric.squeeze()) # take magnitude
    metric = 10*np.log10(metric) # convert to dB
    results.append(metric)
results -= np.max(results) # normalize

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta_scan, results) # MAKE SURE TO USE RADIAN FOR POLAR
ax.set_theta_zero_location('N') # make 0 degrees point up
ax.set_theta_direction(-1) # increase clockwise
ax.set_rlabel_position(30)  # Move grid labels away from other labels
ax.set_thetamin(-90)
ax.set_thetamax(90)
plt.show()
fig.savefig('../_images/doa_music.svg', bbox_inches='tight')