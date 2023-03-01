import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Categorical
import math
import numpy as np

class Encoder(nn.Module):
    def __init__(self, relation_num, emb_size, device, num_layers =1):
        super(Encoder, self).__init__()
        self.emb = nn.Embedding(relation_num+1, emb_size, padding_idx=relation_num)
        
        # rel_pair encoder
        self.hidden_size = emb_size
        self.lstm = nn.LSTM(input_size  = emb_size,
                    hidden_size = self.hidden_size,
                    num_layers  = num_layers,
                    batch_first = True)
        #dense layer
        self.fc = nn.Linear(self.hidden_size, 1) # output a score
        #sigmoid function
        self.sigmoid = nn.Sigmoid()
        
        self.relation_num = relation_num  # 2*n_rel
        self.emb_size = emb_size
        self.device = device
        
        # transformer attention
        self.fc_k = nn.Linear(emb_size, emb_size)
        self.fc_q = nn.Linear(emb_size, emb_size)
        self.fc_v = nn.Linear(emb_size, emb_size)
        
    def reduce_rel_pairs(self, inputs):
        batch_size, seq_len, emb_size = inputs.shape
        
        if seq_len >2:
            rel_pairs = []
            # encode the sliding window with LSTM      
            idx = 0
            while idx< seq_len-1:
                rel_pairs_emb =  inputs[:,idx:idx+2,:]  # window size:2 # [batch_size, 2, emb_size]

                out, (h,c) = self.lstm(rel_pairs_emb)  # h: [1, batch_size, hid_size]
                
                #use the hidden state to represent the rel_pair
                hidden = h.squeeze(0) # [batch_size, hid_size]
                rel_pairs.append(hidden)
            
                idx+=1
            
            rel_pairs = torch.stack(rel_pairs,dim=1) # [batch_size, seq_len-1, hid_size]
            rel_pairs_score =self.fc(rel_pairs).squeeze(-1) # [batch_size, seq_len-1]
            rel_pairs_score = self.sigmoid(rel_pairs_score) # [batch_size, seq_len-1]
            
            selected_rel_pair_idx = torch.argmax(rel_pairs_score, dim=-1) # [batch_size]
            
            full_batch = torch.arange(batch_size).to(self.device)
            selected_rel_pair = rel_pairs[full_batch,selected_rel_pair_idx,:] # [batch_size, hid_size]
            
            selected_rel_pair = selected_rel_pair.unsqueeze(1) # [batch_size, 1, hid_size]
            
            scores = self.transformer_attention(selected_rel_pair) # unnormalized # (batch_size, 1, |R|+1)
            probs = torch.softmax(scores, dim=-1) # normalized
            loss = Categorical(probs=probs).entropy() # (batch_size, 1)
            
            selected_rel_pair = self.weightedAverage(scores, selected_rel_pair) # (batch_size, 1, emb_size)
            selected_rel_pair = selected_rel_pair.squeeze(1) # (batch_size, emb_size)
            
            output = inputs.detach().clone()
            
            zero = torch.zeros(emb_size).to(self.device)
            output[full_batch, selected_rel_pair_idx, :] = selected_rel_pair
            output[full_batch, selected_rel_pair_idx+1, :] = zero

            output = output[output.sum(dim=-1)!=0]
            output = output.reshape(batch_size, -1, emb_size) # (batch_size, seq_len-1, emb_size)
        else:
            out, (h,c) = self.lstm(inputs)  # h: [1, batch_size, hid_size]
            #use the hidden state to represent the rel_pair
            output = h.transpose(0, 1) # [batch_size, 1, hid_size]
            loss = torch.zeros((batch_size,1)).to(self.device)
            
        return output, loss

    def forward(self, inputs):
        # inputs are one-hot tensor of relation # [batch_size, seq_len]
        inputs = self.emb(inputs) # [batch_size, seq_len, emb_size]
        batch_size, seq_len, emb_size = inputs.shape
        L = [inputs]
        loss_list =[]
        idx = 0
        while idx< seq_len-1:
            output, loss = self.reduce_rel_pairs(L[-1])
            L.append(output)
            loss_list.append(loss)
            idx+=1
        
        scores = self.transformer_attention(L[-1]) # unnormalized
        
        # entropy for final prediction
        probs = torch.softmax(scores, dim=-1) # normalized
        loss = Categorical(probs=probs).entropy() # (batch_size, 1)
        loss_list.append(loss)
        loss_tensor = torch.cat(loss_list, dim=-1) # [batch_size, seq_len]
        
        return self.predict_head(scores), loss_tensor

    def transformer_attention(self, inputs):
        batch_size, seq_len, emb_size = inputs.shape
        idx_ = torch.LongTensor(range(self.relation_num)).repeat(batch_size, 1).to(self.device)
        relation_emb = self.emb(idx_) # (batch_size, |R|, emb_size)
        key= torch.cat((relation_emb, inputs), dim=1) # (batch_size, |R|+seq_len, emb_size)

        query = self.fc_q(inputs) # (batch_size, seq_len, emb_size)
        key = self.fc_k(key) # (batch_size, |R|+seq_len, emb_size)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) \
        / math.sqrt(self.emb_size)   # (batch_size, seq_len, |R|+seq_len)
        
        return scores # unnormalized
  
    def weightedAverage(self, scores, inputs):
        """
        Take the prob as attention weights
        Compute the weighted sum of all relation vectors
        """
        
        batch_size, seq_len, _ = scores.shape
        
        mask1 = torch.zeros((batch_size, seq_len, self.relation_num), dtype=torch.bool).to(self.device)
        I = torch.eye(seq_len).to(self.device)
        I = I.reshape((1, seq_len, seq_len))
        I = I.repeat(batch_size, 1, 1) # (batch_size, seq_len, seq_len)
        mask2 = ~I.to(torch.bool)
        mask= torch.cat((mask1, mask2), dim=-1) # (batch_size, seq_len,|R|+seq_len)
        
        scores[mask] = float('-inf')
        prob = torch.softmax(scores, dim=-1) # normalized
        
        idx_ = torch.LongTensor(range(self.relation_num)).repeat(batch_size, 1).to(self.device)
        relation_emb = self.emb(idx_) # (batch_size, |R|, emb_size)
        all_emb = torch.cat((relation_emb, inputs), dim=1) # (batch_size, |R|+seq_len, emb_size)
        
        out = prob @ all_emb # (batch_size, seq_len, emb_size)

        return out
    
    def predict_head(self, prob):
        """
        prob output by the final layer can be used to predict the head relation
        Inputs:
            prob - Tensor of shape [batch_size, 1, |R|+1]
        """

        return prob.squeeze(1) # (batch_size, |R|+1)
        
    
    def get_relation_emb(self, rel):
        return self.emb(rel)
