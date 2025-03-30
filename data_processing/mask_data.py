#%%
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

def viz_df(df, figname):
    """Visualize the gain profile"""
    angs = df['Angle'].to_numpy()
    beams = np.arange(0, 63)
    gains = df.iloc[:, 2:].to_numpy()

    norm_gains = (gains - np.min(gains)) / (np.max(gains) - np.min(gains))

    X, Y = np.meshgrid(beams, angs)

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, norm_gains, cmap='jet', edgecolor='none')

    ax.set_xlabel("Beam")
    ax.set_ylabel("Angle")
    ax.set_zlabel("Gain")
    ax.set_title(figname)
    plt.savefig(figname)

def viz_df_heatmap(df, figname):
    """Visualize the gain profile as a heatmap"""
    angs = df['Angle'].to_numpy()
    beams = np.arange(0, 63)
    gains = df.iloc[:, 2:].to_numpy()
    
    # Normalize gains
    norm_gains = (gains - np.min(gains)) / (np.max(gains) - np.min(gains))
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(norm_gains, xticklabels=beams, yticklabels=angs, cmap='jet', cbar=True)
    
    plt.xlabel("Beam")
    plt.ylabel("Angle")
    plt.title(figname)
    plt.savefig(figname)
    plt.show()

def get_avg_gains():
    scaler = MinMaxScaler() 
    pows = [316, 700, 800, 1778, 10000, 20000] 
    avg_gains = np.zeros((91, 63)) 
    for pow in pows:
        p = str(pow)
        csv_file = 'plot3d/filtered_data_power_' + p + '.csv'
        df = pd.read_csv(csv_file)
        viz_df(df, p)
        gains = df.iloc[:, 2:].to_numpy()
        gains_norm = scaler.fit_transform(gains)
        avg_gains = np.add(avg_gains, gains_norm)
    
    avg_gains = np.divide(avg_gains, len(pows))

    df_avg = pd.DataFrame(
        np.column_stack([df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy(), avg_gains]),
        columns=df.columns
    )
    viz_df(df_avg, 'df_avg.png')
    return df_avg

def single_mask(df, decay_rate=0.05, tolerance=1):
    mask = np.zeros((91, 63))

    A = 62
    B = -90
    C = 0
    norm_factor = np.sqrt(A**2 + B**2)  # Precompute normalization factor

    for i in range(91):
        for j in range(1, 63):
            distance = abs(A * i + B * j + C) / norm_factor
            
            # Ensure distance is explicitly a float
            distance = float(distance)

            # Debugging statement to check `distance` values
            # print(f"i={i}, j={j}, distance={distance}")

            # If the point is nearly on the line, set it exactly to 1
            if distance < tolerance:
                mask[i, j] = 1.0
            else:
                mask[i, j] = np.exp(-decay_rate * distance)
    mask[45, :] = np.zeros(63)
    idx = np.arange(91)
    distances = np.abs(idx - 45)
    mask[:, 0] = np.exp(-decay_rate * distances)
    # Ensure df has the correct number of rows before adding mask
    if df.shape[0] != 91:
        raise ValueError(f"DataFrame row count ({df.shape[0]}) does not match expected size (91).")

    df_mask = pd.DataFrame(
        np.column_stack([df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy(), mask]),
        columns=df.columns
    )

    return df_mask


#mask_df1 = single_mask(df_new)

#viz_df(mask_df1, 'mask1.png')

def apply_mask(df, tolerance=0.5, decay_rate=0.05):
    mask = np.zeros((91, 63))
    # Define line equation parameters: Ax + By + C = 0
    A = 62
    B = -90
    C = 0
    norm_factor = np.sqrt(A**2 + B**2)  # Precompute normalization factor

    #mask[45, :] = np.zeros(63)
    idx = np.arange(91)
    distances = np.abs(idx - 46)
    mask[:, 0] = np.exp(-decay_rate * distances)
    # Compute distances and apply exponential decay
    for i in range(91):
        for j in range(1, 63):
            distance = abs(A * i + B * j + C) / norm_factor
            
            # Ensure distance is explicitly a float
            distance = float(distance)
            if i == 46:
                distance += 10

            if distance < tolerance:
                mask[i, j] = 1.0
            else:
                mask[i, j] = np.exp(-decay_rate * distance)

    avg_gains = df.iloc[:, 2:65].to_numpy()
    filtered_gains = np.multiply(mask, avg_gains)    

    df_filt = pd.DataFrame(
        np.column_stack([df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy(), filtered_gains]),
        columns=df.columns
    )

    return df_filt

#filt_df = apply_mask(df_new)

#viz_df(filt_df, 'filtered_df_champ.png')

#print(filt_df.head)

#filt_df.to_csv("train_gain_prof.csv", index=False)
#df_avg = get_avg_gains()
#df_mask = single_mask(df_avg)
#viz_df_heatmap(df_mask, 'df_mask')
df_avg = get_avg_gains()
viz_df_heatmap(df_avg, 'hveatmap1')
df_filt = apply_mask(df_avg)
viz_df(df_filt, 'filtered_df_champ.png')
viz_df_heatmap(df_filt, 'heatmap2')
df_filt.to_csv("train_gain_prof.csv", index=False)


# %

# %%
