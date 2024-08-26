import os,sys,time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import copy
import pdb
import time

from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler

from datetime import datetime

_dtype_ = torch.float32
_device_ = torch.device("cuda")

# seq_len = 1
# seq_len_forward = 1

class PP_MLP(nn.Module): # Price Pridiction

    def __init__(self, seq_len_f):

        super(PP_MLP, self).__init__() 
        self.layer1 = nn.Linear(14, 20)  
        self.layer2 = nn.Linear(20, 20)
        self.layer3 = nn.Linear(20, 20)
        self.output = nn.Linear(20, seq_len_f)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        
        x = self.relu(self.layer1(x)) 
        x = self.relu(self.layer2(x)) 
        x = self.relu(self.layer3(x)) 
        return self.output(x) 
    
class PP_LSTM(nn.Module):
    def __init__(self, seq_len_f):

        super(PP_LSTM, self).__init__()
        self.hidden_dim = 20
        self.lstm = nn.LSTM(14, self.hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.dense_1 = nn.Linear(self.hidden_dim, 20)
        self.dense_2 = nn.Linear(20, 20)
        self.dense_3 = nn.Linear(20, seq_len_f)
        self.relu = nn.ReLU()

    def forward(self, x):

        lstm_out, _ = self.lstm(x)
        
        x = lstm_out[:, -1, :]
        x = self.dropout(x) 
        x = self.relu(self.dense_1(x))
        x = self.relu(self.dense_2(x))
        x = self.dense_3(x)
        
        return x
        
class H5Dataset(Dataset):
    def __init__(self, file_path, norm_flag):
        self.file_path = file_path
        self.dataset = {}

        with h5py.File(self.file_path, 'r') as file:
            self.dataset['inputs'] = file['inputs'][:]
            self.dataset['outputs'] = file['outputs'][:]
            

        if norm_flag == 'n':
            self.normalization()
        if norm_flag == 'd':
            pass
        
    
    def __len__(self):
        return len(self.dataset['inputs'])

    def __getitem__(self, idx):
        input_data = torch.tensor(self.dataset['inputs'][idx], dtype=torch.float32)
        output_data = torch.tensor(self.dataset['outputs'][idx], dtype=torch.float32)
        return input_data, output_data
    
    def normalization(self):
        inputs = self.dataset['inputs']
        outputs = self.dataset['outputs']
        min_vals_i = inputs.min(axis=0)  
        max_vals_i = inputs.max(axis=0)
        if outputs.ndim != 1:
            min_vals_o = np.min(outputs)  
            max_vals_o = np.max(outputs)   
        else:
            min_vals_o = outputs.min(axis=0)  
            max_vals_o = outputs.max(axis=0)         

        ranges_i = max_vals_i - min_vals_i
        ranges_i[ranges_i == 0] = 1

        if (max_vals_o - min_vals_o)!= 0:
            ranges_o = max_vals_o - min_vals_o
        else:
            ranges_o = 1

        self.dataset['inputs'] = (inputs - min_vals_i) / ranges_i
        self.dataset['outputs'] = ((outputs - min_vals_o) / ranges_o).squeeze()

    def add_indices(self):
        inputs = self.dataset['inputs']
        indices = np.arange(inputs.shape[0]).reshape(-1, 1)
        self.dataset['inputs'] = np.hstack((indices, inputs))

def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def denormalize(dataset, model, input):
    outputs = dataset.dataset['outputs']
    min_vals_o = outputs.min(axis=0)  
    max_vals_o = outputs.max(axis=0)         

    range_o = max_vals_o - min_vals_o

    return (model(input)*range_o)+min_vals_o


def seq_gen(dataset, inputs, seq_len, device):

    inputs_o = torch.zeros(inputs.shape[0], seq_len, inputs.shape[1]-1)

    for i in range(inputs.shape[0]):

        idx_i = int(inputs[i][0].item())
        start_idx = max(0, idx_i - seq_len + 1)
        end_idx = idx_i + 1
        segment = dataset.dataset['inputs'][start_idx:end_idx, 1:]

        first_col = segment[:, 0]
        diff = np.diff(first_col)
        last_increasing_idx = np.where(diff <= 0)[0]
        if len(last_increasing_idx) > 0:
            cut_point = last_increasing_idx[-1] + 1
        else:
            cut_point = 0

        segment[:cut_point, :] = 0

        segment_tensor = torch.from_numpy(segment)

        fill_start = seq_len - (end_idx - start_idx)

        target_size = seq_len - fill_start
        
        if segment_tensor.shape[0] < target_size:
            padding_size = target_size - segment_tensor.shape[0]
            padding = torch.zeros(padding_size, segment_tensor.shape[1])
            segment_tensor = torch.cat([padding, segment_tensor], dim=0)
        
        
        
        inputs_o[i, fill_start:seq_len, :] = segment_tensor

    return inputs_o.to(device)

if __name__ == '__main__':

    seq_len = int(sys.argv[1])
    seq_len_forward = int(sys.argv[2])

    dataset_train = H5Dataset(f'../data/lob_data/lob_data_train_f{seq_len_forward}.h5','n')
    dataset_train_dn = H5Dataset(f'../data/lob_data/lob_data_train_f{seq_len_forward}.h5','d')

    dataset_verif = H5Dataset(f'../data/lob_data/lob_data_verif_f{seq_len_forward}.h5','n')
    dataset_verif_dn = H5Dataset(f'../data/lob_data/lob_data_verif_f{seq_len_forward}.h5','d')

    dataset_test = H5Dataset(f'../data/lob_data/lob_data_test_f{seq_len_forward}.h5','n')
    dataset_test_dn = H5Dataset(f'../data/lob_data/lob_data_test_f{seq_len_forward}.h5','d')

    dataset_train.add_indices()
    dataset_verif.add_indices()
    dataset_test.add_indices()

    dataset_train_dn.add_indices()
    dataset_verif_dn.add_indices()
    dataset_test_dn.add_indices()

    PPLSTM_model = PP_LSTM(seq_len_forward).to(_device_)

    loader_train = DataLoader(dataset_train, batch_size = 4096, shuffle = True)
    loader_verif = DataLoader(dataset_verif, batch_size = 4096, shuffle = True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(PPLSTM_model.parameters(), lr=0.00005, weight_decay=1e-5)

    model_save_path_LSTM = f'PPLSTM_model_r{seq_len}_f{seq_len_forward}.pt'
    optimizer_save_path_LSTM = f'PPLSTM_optimizer_r{seq_len}_f{seq_len_forward}.pt'

    # LSTM Training using Time Window

    print("Start training from scratch.")

    epoch_losses_train = []
    epoch_losses_verif = []

    max_epochs = 1

    patience = 10

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    filename = f'LSTM_r{seq_len}_f{seq_len_forward}_loss_train & verif_{timestamp}.dat'

    filename_t = f'LSTM_r{seq_len}_f{seq_len_forward}_time_cost_{timestamp}.dat'

    with open(filename, 'w') as f:

        best_loss_verif = float('inf')
        patience_counter = 0

        for epoch in range(max_epochs):

            PPLSTM_model.train()

            t_start = time.time()

            running_loss_train = 0.0

            print(f"Start epoch {epoch+1}.\n")

            for batch_idx, (inputs, labels) in enumerate(loader_train):
                
                if seq_len_forward == 1:
                    inputs, labels = seq_gen(dataset_train, inputs, seq_len, _device_), labels.to(_device_).unsqueeze(1)
                else:
                    inputs, labels = seq_gen(dataset_train, inputs, seq_len, _device_), labels.to(_device_)

                optimizer.zero_grad()

                outputs = PPLSTM_model(inputs)

                loss = criterion(outputs, labels)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(PPLSTM_model.parameters(), max_norm = 1.0)

                optimizer.step()

                running_loss_train += loss.item()

                if ((batch_idx+1) % 500 == 0) or batch_idx == 0:

                    print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Batch Loss_train: {loss.item()}")

            print('\n')

            PPLSTM_model.eval()

            with torch.no_grad():

                running_loss_verif = 0.0

                for batch_idx, (inputs, labels) in enumerate(loader_verif):

                    if seq_len_forward == 1:
                        inputs, labels = seq_gen(dataset_verif, inputs, seq_len, _device_), labels.to(_device_).unsqueeze(1)
                    else:
                        inputs, labels = seq_gen(dataset_verif, inputs, seq_len, _device_), labels.to(_device_)
                        
                    outputs = PPLSTM_model(inputs)

                    loss = criterion(outputs, labels)

                    running_loss_verif += loss.item()

            epoch_loss_train = running_loss_train/len(loader_train)
            epoch_loss_verif = running_loss_verif/len(loader_verif)

            epoch_losses_train.append(epoch_loss_train)
            epoch_losses_verif.append(epoch_loss_verif)

            # scheduler_step.step()

            f.write(f'Epoch {epoch+1}, Loss_train: {epoch_loss_train}, Loss_verif: {epoch_loss_verif}\n')
            print(f'Epoch {epoch+1}, Epoch Average Loss_train: {epoch_loss_train}, Epoch Average Loss_verif: {epoch_loss_verif}\n')

            t_end = time.time()
            t_duration = t_end - t_start

            with open(filename_t,'a') as f_t:
                f_t.write(f'Epoch {epoch+1} cost {t_duration} s.\n')

            if epoch_loss_verif < best_loss_verif:
                best_loss_verif = epoch_loss_verif
                patience_counter = 0
                torch.save(PPLSTM_model.state_dict(), model_save_path_LSTM)
                torch.save(optimizer.state_dict(), optimizer_save_path_LSTM)
            else :
                patience_counter += 1

            if patience_counter >= patience:
                print(f'Early stopping triggered at epoch {epoch+1}')
                break
        
        print('\n')

    with open(filename, 'r') as file:
        loss_conv_f = file.readlines()
    # Parsing the data to a DataFrame
    loss_conv = {
        'Epoch': [],
        'Loss_train': [],
        'Loss_verif': []
    }

    # Split each line and extract values
    for line in loss_conv_f:
        parts = line.strip().split(', ')
        epoch = int(parts[0].split(' ')[1])
        loss_train = float(parts[1].split(': ')[1])
        loss_verif = float(parts[2].split(': ')[1])
        
        loss_conv['Epoch'].append(epoch)
        loss_conv['Loss_train'].append(loss_train)
        loss_conv['Loss_verif'].append(loss_verif)

    df = pd.DataFrame(loss_conv)

    plt.figure(figsize=(10, 5))
    plt.plot(df['Epoch'], df['Loss_train'], label='Loss on Train')
    plt.plot(df['Epoch'], df['Loss_verif'], label='Loss on Verif')
    plt.title(f'Loss Convergence Curve - LSTM with Time Window Length {seq_len} back and {seq_len_forward} forward')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(1, 20, 1))
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.ylim(0, max(df['Loss_train']))
    plt.legend()
    plt.grid(True)
    plt.savefig(f'LSTM Loss Convergence r{seq_len}_f{seq_len_forward}.pdf', format='pdf')