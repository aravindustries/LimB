#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

actual_beambook_path = "Antenna_beambook.csv"
ideal_beambook_path = "Antenna_beambook_ideal.csv"

actual_beambook = pd.read_csv(actual_beambook_path, dtype=str)
ideal_beambook = pd.read_csv(ideal_beambook_path, dtype=str)

def hex_to_signed_int(hex_val):
    return int(hex_val, 16) if int(hex_val, 16) < 128 else int(hex_val, 16) - 256

def extract_beam_weights(beambook, beam_index):
    beam_data = beambook.iloc[beam_index, :32]  
    
    I_values = np.array([hex_to_signed_int(beam_data[i]) for i in range(0, 32, 2)])
    Q_values = np.array([hex_to_signed_int(beam_data[i]) for i in range(1, 32, 2)])
    
    complex_weights = I_values + 1j * Q_values
    
    magnitudes = np.abs(complex_weights)
    phases = np.angle(complex_weights, deg=True)  
    
    return complex_weights, magnitudes, phases

beam_index = 48
actual_complex, actual_mags, actual_phases = extract_beam_weights(actual_beambook, beam_index)
ideal_complex, ideal_mags, ideal_phases = extract_beam_weights(ideal_beambook, beam_index)

plt.figure(figsize=(10, 5))
plt.plot(actual_phases, 'ro-', label='Actual Beam Phases')
plt.plot(ideal_phases, 'bo-', label='Ideal Beam Phases')
plt.xlabel('Antenna Element Index')
plt.ylabel('Phase (Degrees)')
plt.legend()
plt.title(f'Phase Comparison for Beam {beam_index}')
plt.grid()
plt.show()

print(f"Beam {beam_index} - Actual Complex Weights:", actual_complex)
print(f"Beam {beam_index} - Ideal Complex Weights:", ideal_complex)

