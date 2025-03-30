import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agile_testbed import Agile


B = 4
angles = np.hstack([[0], np.linspace(-45, 45, 62)])
indexes = np.round(np.linspace(11, 52, B)).astype(int)
agile = Agile(angles[indexes])

for path in ['low_snr', 'mid_high_snr', 'mid_low_snr', 'mid_snr', 'ultra_low_snr']:
    folder = os.path.join("../data/spacial_profiles", path)
    combined_df = pd.DataFrame()

    y_err = np.zeros(91)
    for file in os.listdir(folder):
        fp = os.path.join(folder, file)
        df = pd.read_csv(fp)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    
        X = combined_df[[f"Beam_{i}" for i in indexes]].values

        for i in range(X.shape[0] // 91):
            X_sub = X[i*91:(i+1)*91]
            y_err += np.abs(agile.call(X_sub) - np.arange(-45, 46))

        y_err /= X.shape[0] // 91

    # err = np.abs(y_hat - np.arange(-45, 46))
    plt.plot(np.arange(-45, 46), y_err, label=path)

# plt.plot(np.arange(-45, 46), np.arange(-45, 46), 'k--', label="True Angle")
plt.xlabel("True Angle (°)")
plt.ylabel("Estimated Angle (°)")
plt.title(f"Agile Angle Estimation ( B={B} )")
plt.legend()
plt.show()
