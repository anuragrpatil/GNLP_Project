# coding=utf-8
# Copyleft 2019 project LXRT.

import torch.nn as nn
import torch

from param import args
from lxrt.entry import LXRTEncoder
from lxrt.modeling import BertLayerNorm, GeLU
from models import Attention, DecoderWithAttention

from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM, BertForMaskedLM
# Max length including <bos> and <eos>
MAX_VQA_LENGTH = 20
vocab_size = 30000


class VQAModel(nn.Module):
    def __init__(self, num_answers):
        super().__init__()
        
        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_VQA_LENGTH
        )
        hid_dim = self.lxrt_encoder.dim

        #Build Decoder with Attention
        self.decoder  = DecoderWithAttention(attention_dim=hid_dim , embed_dim=hid_dim  , decoder_dim=hid_dim, vocab_size=vocab_size, features_dim=hid_dim, dropout=0.5)
        
        # VQA Answer heads
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

        self.lstm = nn.LSTM(input_size = hid_dim , hidden_size = hid_dim,
                            num_layers = 1, batch_first = True)
        self.linear = nn.Linear(hid_dim,vocab_size) #vocab size of bert is 30000
        

    def forward(self, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        x,fseq, input_id = self.lxrt_encoder(sent, (feat, pos))
        # x = (batch_size, 768)
        # fseq[1] = (batch_size, 36, 768) image features
        #fseq[0] = (batch_size, 20, 768) caption features
        #print(x.shape)
        #print(fseq[1].shape)
        #print(fseq[0].shape)
        # embed = torch.cat((x.unsqueeze(1), fseq[1][:,:19,:]), dim = 1)
        # lstm_outputs, _ = self.lstm(embed)
        # out = self.linear(lstm_outputs)
        #print(out.shape)

        logit = self.logit_fc(x)

        # output_sentence = []
        
        # for batch in range(x.shape[0]):
        #     output_sentence_batch = []
        #     states = None 
        #     inputs = x[batch].unsqueeze(0)
        #     #print(type(inputs))
        #     for i in range(20):
        #         lstm_outputs2, states = self.lstm(inputs.unsqueeze(1), states)
        #         #print('lstm_outputs2', lstm_outputs2)
        #         #print('states', states)
                
        #         lstm_outputs2 = lstm_outputs2.squeeze(1)
        #         out2 = self.linear(lstm_outputs2)
        #         last_pick = out2.max(1)[1] #gives the index of the max 
        #         output_sentence_batch.append(last_pick.item()) #appends the tokens
        #         inputs = fseq[1][batch,i,:].unsqueeze(0)
        #     #print(output_sentence_batch)
        #     output_sentence.append(output_sentence_batch)
        #     #print(output_sentence_batch) 

        predictions, predictions1,encoded_captions, decode_lengths, sort_ind = self.decoder(fseq[1],input_id)    


        # return logit, out, input_id, output_sentence
        return logit, predictions, input_id, predictions

