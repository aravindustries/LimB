from parse_iq import dataIn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from cnn import *
b = 4
def testbest_toCNN(file=None, aug=True):

    
    data = dataIn(file)

    indices = np.round(np.linspace(11, 52, b)).astype(int)
    data = data[:, indices]

    data = data[:, :, :1024 // b]
    data = data.reshape([91, -1])
    data /= np.abs(data).std(axis=1)[:, np.newaxis]
    data = np.stack([data.imag, data.real], axis=1)

    data_val = data[:, :, -(1024 // b):]
    data_val = data_val.reshape([91, -1])
    data_val /= np.abs(data_val).std(axis=1)[:, np.newaxis]
    data_val = np.stack([data_val.imag, data_val.real], axis=1)

    return data_val

model = CNN(B=b)

data = testbest_toCNN("Training_Beam_IQ.csv")

y = np.arange(-45, 46)

X_tensor = torch.FloatTensor(data).to('mps')
y_tensor = torch.LongTensor(y).to('mps')
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=1024)

X_val_tensor = torch.FloatTensor(data_val).to('mps')
y_val_tensor = torch.LongTensor(y).to('mps')
dataset_val = TensorDataset(X_val_tensor, y_val_tensor)
val_dataloader = DataLoader(dataset_val, batch_size=1024)

model.train(
    dataloader,
    val_dataloader,
    epochs=1000,
)



# data_normalized = data / np.abs(data).reshape(91,-1).std(axis=1)[:, np.newaxis, np.newaxis]


# data_real = data_normalized.real
# data_imag = data_normalized.imag

# data_con = np.concatenate([data_imag, data_real], axis=1)

# def testbed_to_cnn():