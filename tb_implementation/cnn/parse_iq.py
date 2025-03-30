import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch


def dataIn(file="Combined_Beam_IQ.csv"):

    df = pd.read_csv("Combined_Beam_IQ.csv", header=None)

    print(df.head(14))

    df.drop(df.columns[1], axis=1, inplace=True)

    print(df.iloc[45])

    angles = df.iloc[:, 0].values  # First column (angle)
    beams = df.iloc[:, 1].values  # Second column (beam)
    complex_data = df.iloc[
        :, 2:
    ].values  # Complex data values starting from the third column

    # print(angles)
    # print(beams)
    # print(complex_data.shape)

    # Get the unique angles and beams
    unique_angles = np.unique(angles)
    unique_beams = np.unique(beams)
    num_values = complex_data.shape[1]

    # Create a mapping for angles to indices
    angle_map = {angle: idx for idx, angle in enumerate(unique_angles)}

    # Initialize a 3D array with the shape (num_angles, num_beams, num_values)
    result = np.zeros(
        (len(unique_angles), len(unique_beams), num_values), dtype=complex
    )

    # Fill the array with the complex numbers
    for i, angle in enumerate(unique_angles):
        for j, beam in enumerate(unique_beams):
            # Mask to select rows where angle and beam match
            mask = (angles == angle) & (beams == beam)

            # Check if more than one row matches the (angle, beam) pair
            if np.sum(mask) == 1:
                result[i, j, :] = complex_data[mask, :].flatten()
            else:
                # If multiple rows match, you can average them or take the first one
                result[i, j, :] = complex_data[mask, :][
                    0
                ]  # Take the first row that matches

    return result


if __name__ == "__main__":
    dataIn()
