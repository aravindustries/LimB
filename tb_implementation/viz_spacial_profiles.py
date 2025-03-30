#%%
import pandas as pd
import utils2

snrvec = {-5, 0, 5, 10, 15}

for snr in snrvec: 
    df_inf = pd.read_csv('../data_processing/train_gain_prof.csv')
    #df_noisy = pd.read_csv('../data/spacial_profiles/gp_' + str(snr) + '.csv')
    #fin_snr = utils2.get_snr(df_inf, df_noisy)
    #lab = 'SNR: ' + str(fin_snr) + '.png'
    utils2.viz_df(df_inf, 'champ')


# %%
