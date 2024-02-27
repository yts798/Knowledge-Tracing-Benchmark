

from KTStrategy import KTStrategy

import os
from torch.nn import Module, Embedding, Parameter, Sequential, Linear, ReLU, \
    Dropout, MultiheadAttention, GRU, LayerNorm
from torch.nn.functional import one_hot, binary_cross_entropy
from sklearn import metrics
from datareader import reader

import numpy as np
import pandas as pd
import json
import pickle
from torch.nn.init import kaiming_normal_


import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import SGD, Adam

from utils import collate_fn

from torch import nn
from torch.nn import Module, Embedding, Linear, Dropout, MaxPool1d, Sequential, ReLU
import copy
import numpy as np
import torch

if torch.cuda.is_available():
    device = "cuda"
    print("i have cudaaaa!!")
else:
    device = "cpu"
    print("i have CPUUUU!!")


class SAKTM(KTStrategy):
    def __init__(self):
        self.data_name = None
        self.re = -1
        self.model_name = "saktm"

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

        model = SAKT_M(dataset.num_q, **model_config).to(device)
        train_size = int(len(dataset) * train_ratio)
        test_size = len(dataset) - train_size

        train_dataset, test_dataset = random_split(
            dataset, [train_size, test_size], generator=torch.Generator(device=device)
        )

        if os.path.exists(os.path.join(dataset.dataset_dir, "train_indices.pkl")):
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
            opt = SGD(model.parameters(), learning_rate, momentum=0.9)
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




class SAKT_M(Module):
    def __init__(self, num_q, n, d, num_attn_heads, dropout, num_en=2):
        super().__init__()
        self.model_name = "sakt"

        self.num_q = num_q
        self.seq_len = n
        self.emb_size = d
        self.num_attn_heads = num_attn_heads
        self.dropout = dropout
        self.num_en = num_en

        # num_c, seq_len, emb_size, num_attn_heads, dropout, emb_path="")
        self.interaction_emb = Embedding(self.num_q * 2, self.emb_size)
        self.exercise_emb = Embedding(self.num_q, self.emb_size)
        # self.P = Parameter(torch.Tensor(self.seq_len, self.emb_size))
        self.position_emb = Embedding(self.seq_len, self.emb_size)

        self.blocks = get_clones(Blocks(self.emb_size, num_attn_heads, dropout), self.num_en)

        self.dropout_layer = Dropout(dropout)
        self.pred = Linear(self.emb_size, 1)



    def base_emb(self, q, r, qry):
        x = q + self.num_q * r
        qshftemb, xemb = self.exercise_emb(qry), self.interaction_emb(x)

        posemb = self.position_emb(pos_encode(xemb.shape[1]))
        xemb = xemb + posemb
        return qshftemb, xemb

    def forward(self, q, r, qry):
        qemb, qshftemb, xemb = None, None, None
        qshftemb, xemb = self.base_emb(q, r, qry)
        # print(f"qemb: {qemb.shape}, xemb: {xemb.shape}, qshftemb: {qshftemb.shape}")
        for i in range(self.num_en):
            xemb = self.blocks[i](qshftemb, xemb, xemb)

        p = torch.sigmoid(self.pred(self.dropout_layer(xemb))).squeeze(-1)
        return p, xemb



    def train_model(
        self, train_loader, test_loader, num_epochs, opt, ckpt_path
    ):
        '''
            Args:
                train_loader: the PyTorch DataLoader instance for training
                test_loader: the PyTorch DataLoader instance for test
                num_epochs: the number of epochs
                opt: the optimization to train this model
                ckpt_path: the path to save this model's parameters
        '''
        aucs = []
        loss_means = []

        max_auc = 0
        epoch_best = -1
        for i in range(1, num_epochs + 1):
            loss_mean = []
            loss_mean_dspl = []
            for data in train_loader:
                q, r, qshft, rshft, m = data

                self.train()

                p, _ = self(q.long(), r.long(), qshft.long())
                p = torch.masked_select(p, m)
                t = torch.masked_select(rshft, m)

                opt.zero_grad()
                loss = binary_cross_entropy(p, t)
                loss.backward()
                opt.step()
                loss_mean_dspl.append(loss.detach().cpu().numpy().item())
                loss_mean.append(loss.detach().cpu().numpy())

            with torch.no_grad():
                for data in test_loader:
                    q, r, qshft, rshft, m = data

                    self.eval()

                    p, _ = self(q.long(), r.long(), qshft.long())
                    p = torch.masked_select(p, m).detach().cpu()
                    t = torch.masked_select(rshft, m).detach().cpu()

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

class Blocks(Module):
    def __init__(self, emb_size, num_attn_heads, dropout) -> None:
        super().__init__()

        self.attn = MultiheadAttention(emb_size, num_attn_heads, dropout=dropout)
        self.attn_dropout = Dropout(dropout)
        self.attn_layer_norm = LayerNorm(emb_size)

        self.FFN = transformer_FFN(emb_size, dropout)
        self.FFN_dropout = Dropout(dropout)
        self.FFN_layer_norm = LayerNorm(emb_size)

    def forward(self, q=None, k=None, v=None):
        q, k, v = q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)
        # attn -> drop -> skip -> norm
        # transformer: attn -> drop -> skip -> norm transformer default
        causal_mask = ut_mask(seq_len=k.shape[0])
        attn_emb, _ = self.attn(q, k, v, attn_mask=causal_mask)

        attn_emb = self.attn_dropout(attn_emb)
        attn_emb, q = attn_emb.permute(1, 0, 2), q.permute(1, 0, 2)

        attn_emb = self.attn_layer_norm(q + attn_emb)

        emb = self.FFN(attn_emb)
        emb = self.FFN_dropout(emb)
        emb = self.FFN_layer_norm(attn_emb + emb)
        return emb

class transformer_FFN(Module):
    def __init__(self, emb_size, dropout) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.dropout = dropout
        self.FFN = Sequential(
                Linear(self.emb_size, self.emb_size),
                ReLU(),
                Dropout(self.dropout),
                Linear(self.emb_size, self.emb_size),
                # Dropout(self.dropout),
            )
    def forward(self, in_fea):
        return self.FFN(in_fea)

def ut_mask(seq_len):
    """ Upper Triangular Mask
    """
    return torch.triu(torch.ones(seq_len,seq_len),diagonal=1).to(dtype=torch.bool).to(device)

def lt_mask(seq_len):
    """ Upper Triangular Mask
    """
    return torch.tril(torch.ones(seq_len,seq_len),diagonal=-1).to(dtype=torch.bool).to(device)

def pos_encode(seq_len):
    """ position Encoding
    """
    return torch.arange(seq_len).unsqueeze(0).to(device)

def get_clones(module, N):
    """ Cloning nn modules
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

