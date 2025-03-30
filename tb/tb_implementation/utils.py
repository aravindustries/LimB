##%%
#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt

#def add_noise(df, K):
    #angles = df['Angle']
    #power_levels = df['Power_Level']

    #beam_columns = [col for col in df.columns if col.startswith('Beam_')]
    #beam_data = df[beam_columns]
    #noise_std_dev = K * beam_data.mean().mean() 

    #noisy_beam_data = beam_data + np.random.normal(0, noise_std_dev, beam_data.shape)
    #noisy_df = pd.concat([angles, power_levels, noisy_beam_data], axis=1)

    #return noisy_df

#def viz_df(df, figname):
    #angs = df['Angle'].to_numpy()
    #tx_powers = df['Power_Level'].to_numpy()
    #beams = np.arange(0, 63)
    #gains = df.iloc[:, 2:].to_numpy()

    #norm_gains = (gains - np.min(gains)) / (np.max(gains) - np.min(gains))

    #X,Y = np.meshgrid(beams, angs)

    #fig = plt.figure(figsize=(10,7))
    #ax = fig.add_subplot(111, projection='3d')
    #ax.plot_surface(X, Y, norm_gains, cmap='jet', edgecolor='none')

    #ax.set_xlabel("beam")
    #ax.set_ylabel("angle")
    #ax.set_zlabel("Gain")
    #ax.set_title("3D gain plot")
    #plt.savefig(figname)

#def add_multipath(df, K):
    #num_peaks = np.random.randint(3,5)

    #angles = np.linspace(-45, 45, 91)
    #beams = np.arange(0, 63)

    #gain_profile = np.zeros((91, 63))

    #for _ in range(num_peaks):
        #peak_angle = np.random.uniform(-45, 45)
        #peak_beam = np.random.randint(0, 63)
        #peak_gain = np.random.uniform(0, 1)

        #angle_spread = np.random.uniform(2,3)
        #beam_spread = np.random.uniform(2,3)

        #for i, angle in enumerate(angles):
            #for j, beam in enumerate(beams):
                #gain_profile[i, j] += peak_gain * np.exp(
                    #-((angle - peak_angle) ** 2 / (2 * angle_spread ** 2)) 
                    #-((beam - peak_beam) ** 2 / (2 * beam_spread ** 2))
                #)
    
    #print(gain_profile.shape)

    #gains_existing = df.iloc[:, 2:].to_numpy(dtype=float)
    #gains_norm = (gains_existing - np.min(gains_existing)) / (np.max(gains_existing) - np.min(gains_existing))

    #gains_comb = gains_norm + (K * gain_profile)

    #df_combined = pd.DataFrame(
        #np.column_stack([df['Angle'].to_numpy(), df['Power_Level'].to_numpy(), gains_comb]),
        #columns=df.columns
    #)

    #return df_combined

#def run_test():

    #df = pd.read_csv('../data_processing/power_800_data.csv')

    #viz_df(df, 'og_gain_profile.png')

    #ndf = add_noise(df, 0.3)

    #viz_df(ndf, 'noisey_gain_profile.png')

    #mpdf = add_multipath(ndf, 0.5)
    #print(mpdf['Angle'])

    #viz_df(mpdf, 'multipath_gain_profile.png')

#run_test()
# %%
