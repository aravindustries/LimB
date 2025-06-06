import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agile_testbed import Agile

for snr in [-5, 0, 5, 10, 15]:
    df = pd.read_csv(f'../data/spacial_profiles/gp_{snr}.csv')
    df.drop(columns=["Power_Level"], inplace=True)

    max_power_angles = np.zeros(63)

    # Loop through each beam column
    for beam_idx in range(63):
        beam_col = f"Beam_{beam_idx}"
        
        # Find the row index with the maximum power for this beam
        max_power_idx = df[beam_col].idxmax()
        
        # Get the corresponding angle value
        max_power_angles[beam_idx] = df.loc[max_power_idx, "Angle"]

    # Now max_power_angles contains the angle that gives maximum power for each beam
    # The index of the array corresponds to the beam number

    B = 8

    angles = np.hstack([[0], np.linspace(-45, 45, 62)])
    indexes = np.round(np.linspace(11, 52, B)).astype(int)

    agile = Agile(angles[indexes])

<<<<<<< HEAD
    X = df[[f"Beam_{i}" for i in indexes]].values
    print(X.shape)
    y_hat = agile.call(X)

    y = np.arange(-45, 46)

    plt.plot(y, y_hat-y, label=f"Agile (SNR={snr})")

plt.title("Agile Angle Error")
plt.xlabel("True Angle (°)")
plt.ylabel("Estimated Angle Error (°)")
plt.legend()
plt.show()
=======
X = df[[f"Beam_{i}" for i in indexes]].values
print(X.shape)
y_hat = agile.call(X)

# %%
>>>>>>> 034f0d5 (Stashing local changes before rebase)
