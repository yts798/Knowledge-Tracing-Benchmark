
from torch.nn import Module, Embedding, LSTM, Linear, Dropout
from torch.nn.functional import one_hot, binary_cross_entropy

from KTStrategy import KTStrategy

from KTStrategy import KTStrategy

import os
from torch.nn import Module, Embedding, Parameter, Sequential, Linear, \
    Dropout, MultiheadAttention, GRU, LayerNorm, Transformer
from torch.nn.functional import one_hot, binary_cross_entropy
from sklearn import metrics
from datareader import reader
import os

from torch.nn.init import normal_

import numpy as np
import pandas as pd
import json
import pickle

import torch.nn as nn
import torch.nn.functional as F

import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import SGD, Adam

from utils import collate_fn

if torch.cuda.is_available():
    device = "cuda"
    print("i have cudaaaa!!")
else:
    device = "cpu"
    print("i have CPUUUU!!")

class MYSAINT(KTStrategy):
    def __init__(self):
        self.data_name = None
        self.re = -1
        self.model_name = "mysaint"

    def set_data_name(self, data_name):
        self.data_name = data_name

    def show_model(self):
        print('You are using the model', self.model_name, ' with the dataset', self.data_name)

    def set_data_name(self, data_name):
        self.data_name = data_name

    def set_model_name(self, model_name):
        self.model_name = model_name

    def run_model(self):
        if not os.path.isdir("ckpts"):
            os.mkdir("ckpts")

        ckpt_path = os.path.join("ckpts", self.model_name)
        if not os.path.isdir(ckpt_path):
            os.mkdir(ckpt_path)
        print(ckpt_path, self.data_name)
        ckpt_path = os.path.join(ckpt_path, self.data_name)
        if not os.path.isdir(ckpt_path):
            os.mkdir(ckpt_path)

        with open("config_model.json") as f:
            config = json.load(f)
            model_config = config[self.model_name]
            train_config = config["train_config"]

        batch_size = train_config["batch_size"]
        num_epochs = train_config["num_epochs"]
        train_ratio = train_config["train_ratio"]
        learning_rate = train_config["learning_rate"]
        optimizer = train_config["optimizer"]  # can be [sgd, adam]
        seq_len = train_config["seq_len"]

        dataset = reader(seq_len, self.data_name)
        # if self.dataset_name == "ASSIST2009":
        #     dataset = ASSIST2009(seq_len)
        # elif dataset_name == "ASSIST2015":
        #     dataset = ASSIST2015(seq_len)
        # elif dataset_name == "Algebra2005":
        #     dataset = Algebra2005(seq_len)
        # elif dataset_name == "Statics2011":
        #     dataset = Statics2011(seq_len)



        with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
            json.dump(model_config, f, indent=4)
        with open(os.path.join(ckpt_path, "train_config.json"), "w") as f:
            json.dump(train_config, f, indent=4)

        model = PlusSAINTModule(dataset.num_q, **model_config).to(device)
        train_size = int(len(dataset) * train_ratio)
        test_size = len(dataset) - train_size

        train_dataset, test_dataset = random_split(
            dataset, [train_size, test_size], generator=torch.Generator(device=device)
        )

        if False:
        # if os.path.exists(os.path.join(dataset.dataset_dir, "train_indices.pkl")):
            with open(
                    os.path.join(dataset.dataset_dir, "train_indices.pkl"), "rb"
            ) as f:
                train_dataset.indices = pickle.load(f)
            with open(
                    os.path.join(dataset.dataset_dir, "test_indices.pkl"), "rb"
            ) as f:
                test_dataset.indices = pickle.load(f)
        else:
            with open(
                    os.path.join(dataset.dataset_dir, "train_indices.pkl"), "wb"
            ) as f:
                pickle.dump(train_dataset.indices, f)
            with open(
                    os.path.join(dataset.dataset_dir, "test_indices.pkl"), "wb"
            ) as f:
                pickle.dump(test_dataset.indices, f)

        train_loader = DataLoader(

            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn, generator=torch.Generator(device=device)

        )
        test_loader = DataLoader(
            test_dataset, batch_size=test_size, shuffle=True,
            collate_fn=collate_fn, generator=torch.Generator(device=device)
        )

        if optimizer == "sgd":
            opt = SGD(model.parameters(), learning_rate, momentum=0.8)
        elif optimizer == "adam":
            opt = Adam(model.parameters(), learning_rate)

        aucs, loss_means = \
            model.train_model(
                train_loader, test_loader, num_epochs, opt, ckpt_path
            )

        with open(os.path.join(ckpt_path, "aucs.pkl"), "wb") as f:
            pickle.dump(aucs, f)
        with open(os.path.join(ckpt_path, "loss_means.pkl"), "wb") as f:
            pickle.dump(loss_means, f)

        return aucs, loss_means



class PlusSAINTModule(nn.Module):
    def __init__(self, num_q, MAX_SEQ, EMBED_DIMS, ENC_HEADS, DEC_HEADS, NUM_ENCODER, NUM_DECODER, BATCH_SIZE) :
        # n_encoder,n_detotal_responses,seq_len,max_time=300+1
        super().__init__()
        self.num_q = num_q
        self.loss = nn.BCEWithLogitsLoss()
        self.MAX_SEQ = MAX_SEQ
        self.EMBED_DIMS = EMBED_DIMS
        self.ENC_HEADS = ENC_HEADS
        self.DEC_HEADS = DEC_HEADS
        self.NUM_ENCODER = NUM_ENCODER
        self.NUM_DECODER = NUM_DECODER
        self.encoder_layer = StackedNMultiHeadAttention(n_stacks=NUM_ENCODER,
                                                        n_dims=EMBED_DIMS,
                                                        n_heads=ENC_HEADS,
                                                        seq_len=MAX_SEQ,
                                                        n_multihead=1, dropout=0.7)
        self.decoder_layer = StackedNMultiHeadAttention(n_stacks=NUM_DECODER,
                                                        n_dims=EMBED_DIMS,
                                                        n_heads=DEC_HEADS,
                                                        seq_len=MAX_SEQ,
                                                        n_multihead=2, dropout=0.7)
        self.encoder_embedding = EncoderEmbedding(n_exercises=num_q,
                                                  n_dims=EMBED_DIMS, seq_len=MAX_SEQ)
        self.decoder_embedding = DecoderEmbedding(
            n_responses=2, n_dims=EMBED_DIMS, seq_len=MAX_SEQ)
        # self.elapsed_time = nn.Linear(1, EMBED_DIMS)
        self.fc = nn.Linear(EMBED_DIMS, 1)

    def forward(self, x, y):
        enc = self.encoder_embedding(x)
        dec = self.decoder_embedding(y)
        # elapsed_time = x["input_rtime"].unsqueeze(-1).float()
        # # ela_time = self.elapsed_time(elapsed_time)
        # dec = dec + ela_time
        # this encoder
        encoder_output = self.encoder_layer(input_k=enc,
                                            input_q=enc,
                                            input_v=enc)
        # this is decoder
        decoder_output = self.decoder_layer(input_k=dec,
                                            input_q=dec,
                                            input_v=dec,
                                            encoder_output=encoder_output,
                                            break_layer=1)
        # fully connected layer
        out = self.fc(decoder_output)
        out = torch.sigmoid(out)
        return out.squeeze()
#n


    def train_model(self, train_loader, test_loader, num_epochs, opt, ckpt_path):
        aucs = []
        loss_means = []

        max_auc = 0
        epoch_best = -1
        for i in range(1, num_epochs + 1):
            # print("Curr epoch:", i)
            loss_mean = []
            loss_mean_dspl = []

            for data in train_loader:
                # print("1 train:")
                q, r, _, _, m = data

                self.train()

                p = self(q.long(), r.long())
                p = torch.masked_select(p, m)
                # t = torch.masked_select(r, m).float()
                t = torch.masked_select(r, m)

                opt.zero_grad()
                # print(p)
                # print(t)

                loss = binary_cross_entropy(p, t)
                # print("loss", loss)
                loss.backward()
                opt.step()
                loss_mean_dspl.append(loss.detach().cpu().numpy().item())
                loss_mean.append(loss.detach().cpu().numpy())

            with torch.no_grad():
                for data in test_loader:
                    # print("1 test:")
                    q, r, _, _, m = data

                    self.eval()

                    p = self(q.long(), r.long())
                    # print(p)
                    # print(t)
                    p = torch.masked_select(p, m).detach().cpu()
                    # t = torch.masked_select(r, m).float().detach().cpu()
                    t = torch.masked_select(r, m).detach().cpu()

                    auc = metrics.roc_auc_score(
                        y_true=t.numpy(), y_score=p.numpy()
                    )
                    print(loss_mean_dspl)
                    loss_mean_dspl = []
                    loss_mean = np.mean(loss_mean)

                    print(
                        "Epoch: {},   AUC: {},   Loss Mean: {}"
                        .format(i, auc, loss_mean)
                    )

                    if auc > max_auc:
                        epoch_best = i
                        torch.save(
                            self.state_dict(),
                            os.path.join(
                                ckpt_path, "model.ckpt"
                            )
                        )
                        max_auc = auc

                    aucs.append(auc)
                    loss_means.append(loss_mean)
            if i - epoch_best >= 10:
                print("ENOUGH!, The best epoch is", max_auc)
                break
        return aucs, loss_means

#o

    # def train_model(self, train_loader, test_loader, num_epochs, opt, ckpt_path):
    #     aucs = []
    #     loss_means = []
    #
    #     max_auc = 0
    #     epoch_best = -1
    #     for i in range(1, num_epochs + 1):
    #         loss_mean = []
    #
    #         for data in train_loader:
    #             q, r, _, _, m = data
    #
    #             self.train()
    #
    #             p = self(q.long(), r.long())
    #             p = torch.masked_select(p, m)
    #             t = torch.masked_select(r, m).float()
    #
    #             opt.zero_grad()
    #             # print(p)
    #             # print(t)
    #             loss = binary_cross_entropy(p, t)
    #             # loss = self.loss(p, t)
    #             loss.backward()
    #             opt.step()
    #
    #             loss_mean.append(loss.detach().cpu().numpy())
    #
    #
    #         with torch.no_grad():
    #             for data in test_loader:
    #                 q, r, _, _, m = data
    #
    #                 self.eval()
    #
    #                 p = self(q.long(), r.long())
    #                 p = torch.masked_select(p, m)
    #                 p = p.detach().cpu()
    #                 t = torch.masked_select(r, m).float().detach().cpu()
    #
    #                 auc = metrics.roc_auc_score(
    #                     y_true=t.numpy(), y_score=p.numpy()
    #                 )
    #
    #                 loss_mean = np.mean(loss_mean)
    #
    #                 print(
    #                     "Epoch: {},   AUC: {},   Loss Mean: {}"
    #                     .format(i, auc, loss_mean)
    #                 )
    #
    #                 if auc > max_auc:
    #                     epoch_best = i
    #                     torch.save(
    #                         self.state_dict(),
    #                         os.path.join(
    #                             ckpt_path, "model.ckpt"
    #                         )
    #                     )
    #                     max_auc = auc
    #
    #                 aucs.append(auc)
    #                 loss_means.append(loss_mean)
    #         if i - epoch_best >= 10:
    #             print("ENOUGH!, The best epoch is", max_auc)
    #             break
    #     return aucs, loss_means



class FFN(nn.Module):
    def __init__(self, in_feat):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(in_feat, in_feat)
        self.linear2 = nn.Linear(in_feat, in_feat)

    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = self.linear2(out)
        return out

class EncoderEmbedding(nn.Module):
    def __init__(self, n_exercises, n_dims, seq_len):
        super(EncoderEmbedding, self).__init__()
        self.n_dims = n_dims
        self.seq_len = seq_len
        self.exercise_embed = nn.Embedding(n_exercises, n_dims)
        self.position_embed = nn.Embedding(seq_len, n_dims)

    def forward(self, exercises):
        e = self.exercise_embed(exercises)
        seq = torch.arange(self.seq_len, device=device).unsqueeze(0)
        p = self.position_embed(seq)
        return p + e

class DecoderEmbedding(nn.Module):
    def __init__(self, n_responses, n_dims, seq_len):
        super(DecoderEmbedding, self).__init__()
        self.n_dims = n_dims
        self.seq_len = seq_len
        self.response_embed = nn.Embedding(n_responses, n_dims)
        # self.time_embed = nn.Linear(1, n_dims, bias=False)
        self.position_embed = nn.Embedding(seq_len, n_dims)

    def forward(self, responses):
        e = self.response_embed(responses)
        seq = torch.arange(self.seq_len, device=device).unsqueeze(0)
        p = self.position_embed(seq)
        return p + e

class StackedNMultiHeadAttention(nn.Module):
    def __init__(self, n_stacks, n_dims, n_heads, seq_len, n_multihead=1, dropout=0.0):
        super(StackedNMultiHeadAttention, self).__init__()
        self.n_stacks = n_stacks
        self.n_multihead = n_multihead
        self.n_dims = n_dims
        self.norm_layers = nn.LayerNorm(n_dims)
        # n_stacks has n_multiheads each
        self.multihead_layers = nn.ModuleList(
            n_stacks * [nn.ModuleList(n_multihead * [nn.MultiheadAttention(embed_dim=n_dims,
                                                                           num_heads=n_heads,
                                                                            dropout=dropout), ]), ])
        self.ffn = nn.ModuleList(n_stacks * [FFN(n_dims)])
        self.mask = torch.triu(torch.ones(seq_len, seq_len),
                                diagonal=1).to(dtype=torch.bool)

    def forward(self, input_q, input_k, input_v, encoder_output=None, break_layer=None):
        for stack in range(self.n_stacks):
            for multihead in range(self.n_multihead):
                norm_q = self.norm_layers(input_q)
                norm_k = self.norm_layers(input_k)
                norm_v = self.norm_layers(input_v)
                heads_output, _ = self.multihead_layers[stack][multihead](query=norm_q.permute(1, 0, 2),
                                                                          key=norm_k.permute(
                                                                              1, 0, 2),
                                                                          value=norm_v.permute(
                                                                              1, 0, 2),
                                                                          attn_mask=self.mask.to(device))
                heads_output = heads_output.permute(1, 0, 2)
                # assert encoder_output != None and break_layer is not None
                if encoder_output != None and multihead == break_layer:
                    assert break_layer <= multihead, " break layer should be less than multihead layers and postive integer"
                    input_k = encoder_output
                    input_v = encoder_output
                    input_q = input_q + heads_output
                else:
                    input_q = input_q + heads_output
                    input_k = input_k + heads_output
                    input_v = input_v + heads_output
            last_norm = self.norm_layers(heads_output)
            ffn_output = self.ffn[stack](last_norm)
            ffn_output = ffn_output + heads_output
        # after loops = input_q = input_k = input_v
        return ffn_output

