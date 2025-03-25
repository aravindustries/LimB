import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
import utils
import pandas as pd

device = torch.device("cuda")

csv_file = "../../data/aggregated_training_data/agg_data.csv" 
df = pd.read_csv(csv_file, header=None)
idx = 646

print(df.iloc[idx, 0:2].to_numpy())
egy = utils.normalize_minmax(torch.tensor(df.iloc[idx, 2:65].to_numpy()))
print(egy.shape)

ang = torch.deg2rad(torch.tensor(df.iloc[idx, 0], device=device))
print(ang)

X = utils.create_noisy_tone(ang.item(), 0)
y = utils.theta_scan(X, 63)

plt.plot(egy)
plt.plot(y)
plt.savefig('gain profile2.png')
# %%
