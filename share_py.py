import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, accuracy_score
from torch.nn.utils.rnn import pack_padded_sequence
from torch.cuda.amp import GradScaler, autocast
import argparse
import datetime
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from semantichar.utils import all_label_augmentation
import semantichar.data 
from semantichar import imagebind_model
from semantichar.imagebind_model import ModalityType
import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
import json

def DataBatch(data, label, text, l, batchsize, shuffle=True):
    
    n = data.shape[0]
    print( n )
    if shuffle:
        index = np.random.permutation(n)
    else:
        index = np.arange(n)
    for i in range(int(np.ceil(n/batchsize))):
        inds = index[i*batchsize : min(n,(i+1)*batchsize)]
        yield inds, data[inds], label[inds], text[inds], l[inds]
        
def trainer(opt, 
            enc, 
            dec, 
            cross_entropy, 
            optimizer, 
            tr_data, 
            tr_label, 
            tr_text, 
            len_text, 
            break_step, 
            vocab_size, 
            device
):
    """
    Train the model.
    Args:
        opt: user-specified configurations.
        enc: encoder of the model.
        dec: decoder of the model.
        cross_entropy: loss function.
        optimizer: optimizer (default is Adam).
        tr_data, tr_label, tr_text, len_text: training data, label, label sequence, length of the label sequence. 
        break_step: length of the longest label sequence length (i.e., maximum decoding step).
        vocab_size: label name vocabulary size.
        device: cuda or cpu.
    """

    enc.train()
    dec.train()  

    total_loss = 0
    for inds,batch_data, batch_label, batch_text, batch_len in \
        DataBatch(tr_data, tr_label, tr_text, len_text, opt['batchSize'],shuffle=True):
        
        batch_text = all_label_augmentation(batch_text, opt['prob'], break_step, vocab_size)

        batch_data = batch_data.to(device)
        batch_label = batch_label.to(device)
        batch_text = batch_text.to(device)
        batch_len = batch_len.to(device)

        enc_hidden = enc(batch_data)
        pred, batch_text_sorted, decode_lengths, sort_ind \
            = dec(enc_hidden, batch_text, batch_len)
        
        targets = batch_text_sorted[:, 1:]

        pred, *_ = pack_padded_sequence(pred, decode_lengths, batch_first=True)
        targets, *_ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        loss = cross_entropy(pred, targets.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += len(batch_data) * loss.item()

    total_loss /= len(tr_data)
    
    return total_loss 




    






from semantichar.dataset import prepare_dataset


class Encoder(nn.Module):

    def __init__(self,
                 d_input: int,
                 d_model: int,
                 d_output: int,
                 seq_len: int):
        super().__init__()

        self.layer1 = nn.Conv1d(in_channels=d_input, out_channels=d_model, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.act1 = nn.ReLU()

        self.layer2 = nn.Conv1d(in_channels=d_model, out_channels=d_output, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.act2 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b,t,c = x.size()

        out = self.layer1(x.permute(0,2,1))
        #out = self.act1((out))
        out = self.act1(self.bn1(out))

        out = self.layer2(out)
        #out = self.act2(out)
        out = self.act2(self.bn2(out)) # (b, d_output, seq_len)

        return out.permute(0,2,1) # (b, seq_len, d_output)
        


    
class Decoder(nn.Module):
    
    def __init__(self, embed_dim, decoder_dim, vocab, encoder_dim, device, dropout=0.5):
      
        super(Decoder, self).__init__()

        self.encoder_dim = encoder_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.dropout = dropout
        self.device = device

        self.embedding = nn.Embedding(self.vocab_size, embed_dim)  
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim, decoder_dim, bias=True)  
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  
        self.fc = nn.Linear(decoder_dim, self.vocab_size) 
        self.load_pretrained_embeddings()

    def load_pretrained_embeddings(self):

        inputs = {
            ModalityType.TEXT: semantichar.data.load_and_transform_text(self.vocab, self.device)
        }
        model = imagebind_model.imagebind_huge(pretrained=True)
        model.eval()
        model.to(self.device)
        with torch.no_grad():
            embeddings = model(inputs)['text']
        self.embedding.weight = nn.Parameter(embeddings)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        
        mean_encoder_out = encoder_out.mean(dim=1) 
        h = self.init_h(mean_encoder_out) 
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  
        seq_len = encoder_out.size(1)

        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        embeddings = self.embedding(encoded_captions.long()) 

        h, c = self.init_hidden_state(encoder_out)  

        decode_lengths = (caption_lengths - 1).tolist()
        
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(self.device)

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths]) 
            h, c = self.decode_step(embeddings[:batch_size_t, t, :], \
                (h[:batch_size_t], c[:batch_size_t]))
            preds = self.fc(self.dropout(h))  
            predictions[:batch_size_t, t, :] = preds
        return predictions, encoded_captions, decode_lengths, sort_ind
    
os.chdir('..')



# Updated values for the arguments
dataset = 'easy_imu_phone'
data_path = './'
manualSeed = 2023
epochs = 150
early_stopping = 50
batchSize = 16
lr = 1e-4
prob = 0.4
cuda = True
run_tag = 'test'
model_path = './model/'

# Print the updated values
print(f'dataset: {dataset}')
print(f'data_path: {data_path}')
print(f'manualSeed: {manualSeed}')
print(f'epochs: {epochs}')
print(f'early_stopping: {early_stopping}')
print(f'batchSize: {batchSize}')
print(f'lr: {lr}')
print(f'prob: {prob}')
print(f'cuda: {cuda}')
print(f'run_tag: {run_tag}')
print(f'model_path: {model_path}')

# Set random seed
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
np.random.seed(manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not cuda:
    print("You have a cuda device, so you might want to run with --cuda as option")
device = torch.device("cuda:0" if cuda else "cpu")

current_directory = os.getcwd()
print("Current Directory:", current_directory)

data_root = data_path + '/dataset/' + dataset
config_file = data_path + '/configs/' + dataset + '.json'
with open(config_file, 'r') as config_file:
    data = json.load(config_file)
    label_dictionary = {int(k): v for k, v in data['label_dictionary'].items()}

tr_data = np.load(data_root + '/x_train.npy')
tr_label = np.load(data_root + '/y_train.npy')

test_data = np.load(data_root + '/x_test.npy')
test_label = np.load(data_root + '/y_test.npy')

seq_len, dim, class_num, vocab_size, break_step, word_list, pred_dict, seqs, \
    tr_data, test_data, \
    tr_label, test_label, \
    tr_text, test_text, \
    len_text, test_len_text = prepare_dataset(tr_data, tr_label, test_data, test_label, label_dictionary)

config = {
    'batchSize': batchSize,
    'epochs': epochs,
    'run_tag': run_tag,
    'dataset': dataset,
    'cuda': cuda,
    'manualSeed': manualSeed,
    'data_path': data_path,
    'early_stopping': early_stopping,
    'lr': 0.0001,
    'prob': prob,
    'model_path': model_path
}
device = torch.device('cpu')

enc = Encoder(d_input=dim, d_model=128, d_output=128, seq_len=seq_len)
dec = Decoder(embed_dim=1024, decoder_dim=128, vocab=word_list, encoder_dim=128, device=device)

# Assuming 'config' is a dictionary containing 'run_tag'
enc_load_path = os.path.join('model', f"{config['run_tag']}_enc.pth")
dec_load_path = os.path.join('model', f"{config['run_tag']}_dec.pth")



# Load the state dictionary from the file
enc_state_dict = torch.load(enc_load_path)
dec_state_dict = torch.load(dec_load_path)

# Load the state dictionary into the model
enc.load_state_dict(enc_state_dict)
dec.load_state_dict(dec_state_dict)

# Ensure the models are on the correct device
enc.to(device)
dec.to(device)

enc.eval()
dec.eval()

hypotheses = list()
batch_size = test_data.size(0)
pred_whole = torch.zeros_like(test_label)
seqs = seqs.to(device)
#print(seqs)
config['batchSize'] = 16
total_evaluation_time = 0  # Initialize total evaluation time
total_samples = 0  # Initialize total number of samples
with torch.no_grad():
    for batch_idx, (inds, batch_data, batch_label, batch_text, batch_len) in enumerate(
            DataBatch(test_data, test_label, test_text, test_len_text, config['batchSize'] , shuffle=True)
        ):
            #pred_whole = torch.zeros_like(test_label)
            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device)
            batch_text = batch_text.to(device)
            batch_len = batch_len.to(device)
            #print( batch_idx )
            #print( inds)
            

            batch_size = batch_data.size(0)
            
            total_samples += batch_size  # Accumulate the number of samples
            start_time = time.time()
            #print(enc.layer1.weight.device)
            #print(batch_data.device)
            #print(dec.init_h.weight.device)
            encoder_out = enc(batch_data)  # (batch_size, enc_seq_len, encoder_dim)
            #print(encoder_out.shape)
            enc_seq_len = encoder_out.size(1)
            encoder_dim = encoder_out.size(2)

            encoder_out = encoder_out.unsqueeze(1).expand(batch_size, class_num, enc_seq_len, encoder_dim)
            encoder_out = encoder_out.reshape(batch_size * class_num, enc_seq_len, encoder_dim)
            #print(seqs)
            k_prev_words = seqs[:, 0].unsqueeze(0).expand(batch_size, class_num).long()  # (batch_size, class_num)
            #print(k_prev_words)
            k_prev_words = k_prev_words.reshape(batch_size * class_num, 1)  # (batch_size * class_num, 1)
            
            h, c = dec.init_hidden_state(encoder_out)

            seq_scores = torch.zeros((batch_size, class_num)).to(device)

            for step in range(1, break_step):
                embeddings = dec.embedding(k_prev_words).squeeze(1)  # (batch_size * class_num, embed_dim)
                h, c = dec.decode_step(embeddings, (h, c))
                scores = dec.fc(h.reshape(batch_size, class_num, -1))  # (batch_size, class_num, vocab_size)
                scores = F.log_softmax(scores, dim=-1)
                k_prev_words = seqs[:, step].unsqueeze(0).expand(batch_size, class_num).long()
                for batch_i in range(batch_size):
                    for class_i in range(class_num):
                        if k_prev_words[batch_i, class_i] != 0:
                            seq_scores[batch_i, class_i] += scores[batch_i, class_i, k_prev_words[batch_i, class_i]]
                k_prev_words = k_prev_words.reshape(batch_size * class_num, 1)  # (batch_size * class_num, 1)
            #print( seq_scores )
            max_indices = seq_scores.argmax(dim=1)
            for batch_i in range(batch_size):
                max_i = max_indices[batch_i]
                seq = seqs[max_i].tolist()
                hypotheses.append([w for w in seq if w not in {0, vocab_size - 1}])
                print(batch_i + batch_idx * config['batchSize'])
                #pred_whole[batch_i + batch_idx * config['batchSize']] = pred_dict["#".join(map(str, hypotheses[-1]))]
                #print(batch_i + batch_idx * config['batchSize'])
                #print
                pred_whole[inds[batch_i]] = pred_dict["#".join(map(str, hypotheses[-1]))]
                #print(test_label[inds[batch_i]])
                #print( pred_whole[inds[batch_i]] )
            #print( pred_whole.shape)
            
            end_time = time.time()
            batch_evaluation_time = end_time - start_time  # Calculate batch evaluation time
            total_evaluation_time += batch_evaluation_time  # Accumulate total evaluation time

            #acc = accuracy_score(test_label.cpu().numpy(), pred_whole.cpu().numpy())
            #prec = precision_score(test_label.cpu().numpy(), pred_whole.cpu().numpy(), average='macro', zero_division=0)
            #rec = recall_score(test_label.cpu().numpy(), pred_whole.cpu().numpy(), average='macro', zero_division=0)
            #f1 = f1_score(test_label.cpu().numpy(), pred_whole.cpu().numpy(), average='macro', zero_division=0)

            #print(f'Total Evaluation Time: {total_evaluation_time:.2f} seconds')
    
            # Calculate evaluation time per batch and per sample
            #eval_time_per_batch = total_evaluation_time / (batch_idx + 1)
            #eval_time_per_sample = total_evaluation_time / total_samples

            #print(f'Average Evaluation Time per Batch: {eval_time_per_batch:.2f} seconds')
            #print(f'Average Evaluation Time per Sample: {eval_time_per_sample:.6f} seconds')
            #print('Test Acc: %.4f Macro-Prec: %.4f Macro-Rec: %.4f Macro-F1: %.4f' % (acc, prec, rec, f1))

            #print(f'Batch {batch_idx + 1} Evaluation Time: {batch_evaluation_time:.2f} seconds')
#print( pred_whole.shape )
#print( test_label.shape )

acc = accuracy_score(test_label.cpu().numpy(), pred_whole.cpu().numpy())
prec = precision_score(test_label.cpu().numpy(), pred_whole.cpu().numpy(), average='macro', zero_division=0)
rec = recall_score(test_label.cpu().numpy(), pred_whole.cpu().numpy(), average='macro', zero_division=0)
f1 = f1_score(test_label.cpu().numpy(), pred_whole.cpu().numpy(), average='macro', zero_division=0)

#print(f'Total Evaluation Time: {total_evaluation_time:.2f} seconds')
    
    # Calculate evaluation time per batch and per sample
#eval_time_per_batch = total_evaluation_time / (batch_idx + 1)
#eval_time_per_sample = total_evaluation_time / total_samples

#print(f'Average Evaluation Time per Batch: {eval_time_per_batch:.2f} seconds')
#print(f'Average Evaluation Time per Sample: {eval_time_per_sample:.6f} seconds')
print('Test Acc: %.4f Macro-Prec: %.4f Macro-Rec: %.4f Macro-F1: %.4f' % (acc, prec, rec, f1))