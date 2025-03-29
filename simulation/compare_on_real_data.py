import numpy as np
import pandas as pd

from agile_testbed import Agile

df = pd.read_csv('../data/spacial_profiles/gp_10.csv')
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

X = df[[f"Beam_{i}" for i in indexes]].values
print(X.shape)
y_hat = agile.call(X)

# %%
