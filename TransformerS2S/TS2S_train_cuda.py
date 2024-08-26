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
from tqdm import trange, tqdm

from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler

torch.backends.cuda.matmul.allow_tf32 = True

from datetime import datetime

_dtype_ = torch.float32
_device_ = torch.device('cuda')

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
    
class PP_LSTMSeq2Seq(nn.Module):
    def __init__(self):

        super(PP_LSTMSeq2Seq, self).__init__()

        emb_size = 64

        self.embedding_encoder = nn.Linear(14, emb_size)
        self.embedding_decoder = nn.Linear(1, emb_size)

        self.mlp_emb = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.LayerNorm(emb_size),
            nn.ELU(),
            nn.Linear(emb_size, emb_size)
        )

        self.hidden_dim = 20

        self.lstm = nn.LSTM(
            input_size=emb_size,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.25
        )

        self.dense_1 = nn.Linear(self.hidden_dim, 10)
        self.dense_2 = nn.Linear(10, 5)
        self.dense_3 = nn.Linear(5, 1)
        self.relu = nn.ReLU()

    def forward(self, x):

        if x.shape[-1] == 1:
            x = self.embedding_decoder(x)
        else:
            x = self.embedding_encoder(x)
        
        x = self.mlp_emb(x)

        lstm_out, _ = self.lstm(x)

        x = lstm_out#[:, -1, :]
        x = self.relu(self.dense_1(x))
        x = self.relu(self.dense_2(x))
        x = self.dense_3(x)
        
        return x

# Components of transformer

class SinPosEncoding(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim

    def forward(self, x):
        device = x.device
        half_dim = self.emb_dim // 2
        pos_emb = np.log(10000) / (half_dim - 1)
        pos_emb = torch.exp(torch.arange(half_dim, device = device) * (- pos_emb))
        pos_emb = x[:, None] * pos_emb[None, :]
        pos_emb = torch.cat((pos_emb.sin(), pos_emb.cos()), dim = -1)
        return pos_emb
    
class AttentionLayer(nn.Module):
    def __init__(self, 
                 hidden_size = 64, 
                 num_heads = 4, 
                 masking = True):
        
        super().__init__()
        self.masking = masking
        self.mh_attn = nn.MultiheadAttention(hidden_size,
                                             num_heads = num_heads,
                                             batch_first = True,
                                             dropout = 0.25)

    def forward(self, 
                x_in, 
                kv_in, 
                key_mask = None):
        
        if self.masking:
            bs, l, h = x_in.shape
            mask = torch.triu(torch.ones(l, l, device=x_in.device), 1).bool()
        else:
            mask = None

        return self.mh_attn(x_in, 
                            kv_in, 
                            kv_in, 
                            attn_mask = mask, 
                            key_padding_mask = key_mask)[0]

class CoreLayer(nn.Module):
    def __init__(self, 
                 hidden_size = 64, 
                 num_heads = 4, 
                 block_type = 'encoder', 
                 masking = True):
        super().__init__()
        self.block_type = block_type

        self.layer_norm_1 = nn.LayerNorm(hidden_size)

        self.attn_1 = AttentionLayer(hidden_size = hidden_size, 
                                     num_heads = num_heads, 
                                     masking = masking)
        
        if self.block_type == 'decoder':
            self.layer_norm_2 = nn.LayerNorm(hidden_size)
            self.attn_2 = AttentionLayer(hidden_size = hidden_size, 
                                         num_heads = num_heads, 
                                         masking = False)
        
        self.layer_norm_mlp = nn.LayerNorm(hidden_size)

        self.mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size * 4),
                                 nn.ELU(),
                                 nn.Linear(hidden_size * 4, hidden_size))
                
    def forward(self,
                x, 
                input_key_mask = None, 
                cross_key_mask = None, 
                kv_cross = None):

        x = self.attn_1(x, x, key_mask = input_key_mask) + x
        x = self.layer_norm_1(x)

        if self.block_type == 'decoder':
            x = self.attn_2(x, kv_cross, key_mask = cross_key_mask) + x
            x = self.layer_norm_2(x)

        x = self.mlp(x) + x
        return self.layer_norm_mlp(x)

class Encoder(nn.Module):
    def __init__(self, 
                 dim_i, 
                 hidden_size = 64, 
                 num_layers = 2, 
                 num_heads = 4):
        super().__init__()

        self.embedding = nn.Linear(dim_i, hidden_size)
        # Instantiate the positional encoding
        self.pos_encoding = SinPosEncoding(hidden_size)

        self.core_layer = nn.ModuleList([
            CoreLayer(hidden_size, 
                      num_heads, 
                      'encoder', 
                      masking = False) for _ in range(num_layers)
        ])
                
    def forward(self, 
                input_seq, 
                padding_mask = None):        
        
        input_embs = self.embedding(input_seq)
        bs, l, h = input_embs.shape

        indices = torch.arange(l, device = input_seq.device)
        pos_encoding = self.pos_encoding(indices).reshape(1, l, h).expand(bs, l, h)
        embs = input_embs + pos_encoding
        
        for layer in self.core_layer:
            embs = layer(embs, 
                         input_key_mask = padding_mask)
        
        return embs
    
class Decoder(nn.Module):
    def __init__(self, 
                 dim_o, 
                 hidden_size = 64, 
                 num_layers = 2, 
                 num_heads = 4,
                 categorical = False):
        super().__init__()

        if categorical:
            self.embedding = nn.Embedding(dim_o, hidden_size)
            self.embedding.weight.data = 0.001 * self.embedding.weight.data
        else:
            self.embedding = nn.Linear(dim_o, hidden_size)

        self.pos_encoding = SinPosEncoding(hidden_size)
        
        self.core_layer = nn.ModuleList([
            CoreLayer(hidden_size, 
                      num_heads, 
                      'decoder', 
                      masking = True) for _ in range(num_layers)
        ])
                
        self.fc_out = nn.Linear(hidden_size, dim_o)
        
    def forward(self, 
                input_seq, 
                encoder_output, 
                input_padding_mask = None, 
                encoder_padding_mask = None):

        input_embs = self.embedding(input_seq)

        bs, l, h = input_embs.shape

        seq_index = torch.arange(l, device = input_seq.device)
        pos_encoding = self.pos_encoding(seq_index).reshape(1, l, h).expand(bs, l, h)
        embs = input_embs + pos_encoding
        
        for layer in self.core_layer:
            embs = layer(embs,
                         input_key_mask = input_padding_mask,
                         cross_key_mask = encoder_padding_mask, 
                         kv_cross = encoder_output)
        
        return self.fc_out(embs)
    
class Transformer(nn.Module):
    def __init__(self,
                 hidden_size = 64, 
                 num_layers = (2,2), 
                 num_heads = 4,
                 categorical = False):
        super().__init__()
        
        self.encoder = Encoder(dim_i=14, 
                               hidden_size = hidden_size, 
                               num_layers = num_layers[0], 
                               num_heads = num_heads,)
        
        self.decoder = Decoder(dim_o = 1, 
                               hidden_size = hidden_size, 
                               num_layers = num_layers[1], 
                               num_heads = num_heads,
                               categorical = categorical)

    def forward(self, input_seq, target_seq):
        
        # input_key_mask = input_seq == 0
        input_key_mask = None
        # output_key_mask = target_seq == 0
        output_key_mask = None
        
        encoder_output = self.encoder(input_seq = input_seq,
                                      padding_mask = input_key_mask)
        
        decoder_output = self.decoder(input_seq = target_seq,
                                      encoder_output = encoder_output,
                                      input_padding_mask = output_key_mask,
                                      encoder_padding_mask = input_key_mask)

        return decoder_output
    
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

    def add_SOS(self):
        outputs = self.dataset['outputs']
        if len(outputs.shape) == 1:
            self.dataset['outputs'] = outputs[:,None]

        self.dataset['outputs'] = np.concatenate((self.dataset['inputs'][:,-1][:,None],self.dataset['outputs']),axis = -1)


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
    categorical = False
    data_dir = '/exports/eddie/scratch/s2601126/lob_data/'

    dataset_train = H5Dataset(f'{data_dir}lob_data_train_f{seq_len_forward}.h5','n')

    dataset_verif = H5Dataset(f'{data_dir}lob_data_verif_f{seq_len_forward}.h5','n')

    dataset_train.add_SOS()
    dataset_verif.add_SOS()


    dataset_train.add_indices()
    dataset_verif.add_indices()


    PPT_model = Transformer(hidden_size = 64,
                            num_layers = (2,2),
                            num_heads = 4,
                            categorical=categorical).to(_device_)
    
    if torch.cuda.device_count() > 1:
        PPLSTM_model = nn.DataParallel(PPT_model)

    scaler = torch.cuda.amp.GradScaler()
    
    batch_size = 512

    loader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle = True)
    loader_verif = DataLoader(dataset_verif, batch_size = batch_size, shuffle = True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(PPT_model.parameters(), lr=1e-4, weight_decay=1e-5)

    cat_str = 'cat' if categorical else 'non_cat'

    model_save_path_Transformer = f'PPTS2S_{cat_str}_model_r{seq_len}_f{seq_len_forward}.pt'
    optimizer_save_path_Transformer = f'PPTS2S_{cat_str}_optimizer_r{seq_len}_f{seq_len_forward}.pt'

    num_model_params = 0
    for param in PPT_model.parameters():
        num_model_params += param.flatten().shape[0]

    print("-This Model has %d (approximately %d kilo) parameters!" % (num_model_params, num_model_params//1e3))

    # Transformer Training using Time Window

    print(f"{seq_len} to {seq_len_forward} Model. Start training from scratch.")

    epoch_losses_train = []
    epoch_losses_verif = []

    max_epochs = 50

    patience = 5

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    filename = f'PPTS2S_{cat_str}_r{seq_len}_f{seq_len_forward}_loss_train & verif_{timestamp}.dat'

    filename_t = f'PPTS2S_{cat_str}_r{seq_len}_f{seq_len_forward}_time_cost_{timestamp}.dat'


    best_loss_verif = float('inf')
    patience_counter = 0

    for epoch in trange(0, max_epochs, leave=False, desc="Epoch"):

        PPT_model.train()

        t_start = time.time()

        running_loss_train = 0.0

        tqdm.write(f"Start epoch {epoch+1}.")

        for batch_idx, (inputs, labels) in enumerate(tqdm(loader_train, desc="Training", leave=False)):
        # for batch_idx, (inputs, labels) in enumerate(loader_train):
            
            # if seq_len_forward == 1:
            #     inputs, labels = seq_gen(dataset_train, inputs, seq_len, _device_), labels.to(_device_).unsqueeze(1).unsqueeze(2)
            # else:
            inputs, labels = seq_gen(dataset_train, inputs, seq_len, _device_), labels.to(_device_).unsqueeze(2)

            decoder_inputs = labels[:,:-1,:]

            with torch.cuda.amp.autocast():

                decoder_outputs = PPT_model(inputs, decoder_inputs)

                loss = criterion(decoder_outputs, labels[:,1:,:])

            optimizer.zero_grad()

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            torch.nn.utils.clip_grad_norm_(PPT_model.parameters(), max_norm = 1.0)

            running_loss_train += loss.item()

            if ((batch_idx+1) % (500*(4096//batch_size)) == 0) or batch_idx == 0:

                tqdm.write(f"Epoch {epoch+1}, Batch {batch_idx+1}, Batch Loss_train: {loss.item()}")

        PPT_model.eval()

        with torch.no_grad():

            running_loss_verif = 0.0

            for batch_idx, (inputs, labels) in enumerate(tqdm(loader_verif, desc="Verification", leave=False)):
            # for batch_idx, (inputs, labels) in enumerate(loader_verif):

                # if seq_len_forward == 1:
                #     inputs, labels = seq_gen(dataset_verif, inputs, seq_len, _device_), labels.to(_device_).unsqueeze(1).unsqueeze(2)
                # else:
                inputs, labels = seq_gen(dataset_verif, inputs, seq_len, _device_), labels.to(_device_).unsqueeze(2)

                decoder_inputs = labels[:,:-1,:]

                with torch.cuda.amp.autocast():

                    decoder_outputs = PPT_model(inputs, decoder_inputs)

                    loss = criterion(decoder_outputs, labels[:,1:,:])

                running_loss_verif += loss.item()

        epoch_loss_train = running_loss_train/len(loader_train)
        epoch_loss_verif = running_loss_verif/len(loader_verif)

        epoch_losses_train.append(epoch_loss_train)
        epoch_losses_verif.append(epoch_loss_verif)

        # scheduler_step.step()
        with open(filename, 'a') as f:

            f.write(f'Epoch {epoch+1}, Loss_train: {epoch_loss_train}, Loss_verif: {epoch_loss_verif}\n')
        tqdm.write(f'Epoch {epoch+1}, Epoch Average Loss_train: {epoch_loss_train}, Epoch Average Loss_verif: {epoch_loss_verif}')

        t_end = time.time()
        t_duration = t_end - t_start

        with open(filename_t,'a') as f_t:
            f_t.write(f'Epoch {epoch+1} cost {t_duration} s.\n')

        if epoch_loss_verif < best_loss_verif:
            best_loss_verif = epoch_loss_verif
            patience_counter = 0
            torch.save(PPT_model.state_dict(), model_save_path_Transformer)
            torch.save(optimizer.state_dict(), optimizer_save_path_Transformer)
        else :
            patience_counter += 1

        if patience_counter >= patience:
            tqdm.write(f'Early stopping triggered at epoch {epoch+1}')
            break


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
    plt.title(f'Loss Convergence Curve - Transformer_Seq2Seq trained with Time Window Length {seq_len} back and {seq_len_forward} forward')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(1, 20, 1))
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.ylim(0, max(df['Loss_train']))
    plt.legend()
    plt.grid(True)
    plt.savefig(f'Transformer_S2S Loss Convergence r{seq_len}_f{seq_len_forward}.pdf', format='pdf')