#%%
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import utils2
import pandas as pd
import mlp2
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df_train = pd.read_csv('../data_processing/train_gain_prof.csv')
df_test = pd.read_csv('../data/spacial_profiles/gp_10.csv')

n_beams = np.array([4])
num_classes = 91  # Adjust based on desired resolution

for n in n_beams:
    dmlp = mlp2.doaMLPClassifier(n, num_classes)
    y_test = df_test['Angle'].to_numpy()
    y_test = np.clip(y_test, -45, 45)
    y_test = np.digitize(y_test, bins=np.linspace(-45, 45, num_classes)) - 1  # Convert to class index
    
    beta = mlp2.get_beams(n, 50)
    scaler = MinMaxScaler()
    X_test = scaler.fit_transform(df_test.iloc[:, 2:65].iloc[:, beta])

    dmlp.iterative_train(df_train, scaler, 100)

    result = dmlp.eval_model(X_test, y_test)
    print("Eval Model Output:", result)
    # Evaluate model and get predictions
    y_actual, y_pred = dmlp.eval_model(X_test, y_test)
    
    # Plot actual vs predicted
    dmlp.plot_predictions(y_actual, y_pred)

# %%
