from torch.nn import Module, Embedding, LSTM, Linear, Dropout
from torch.nn.functional import one_hot, binary_cross_entropy

from KTStrategy import KTStrategy

from KTStrategy import KTStrategy

import os
from torch.nn import Module, Embedding, Parameter, Sequential, Linear, ReLU, \
    Dropout, MultiheadAttention, GRU, LayerNorm
from torch.nn.functional import one_hot, binary_cross_entropy
from sklearn import metrics
from datareaderp1 import readerp1

import numpy as np
import pandas as pd
import json
import pickle

import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import SGD, Adam

from utils import collate_fn, collate_fn_p1, collate_fn_p2

if torch.cuda.is_available():
    device = "cuda"
    print("i have cudaaaa!!")
else:
    device = "cpu"
    print("i have CPUUUU!!")

# means model name + plus n feature, cat first or emb first
class DKT_P1_C(KTStrategy):
    def __init__(self):
        self.data_name = None
        self.re = -1
        self.model_name = "dkt"

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

        dataset = readerp1(seq_len, self.data_name)
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

        model = DKT_Model(dataset.num_q, dataset.num_a, **model_config).to(device)
        train_size = int(len(dataset) * train_ratio)
        test_size = len(dataset) - train_size

        train_dataset, test_dataset = random_split(
            dataset, [train_size, test_size], generator=torch.Generator(device='cuda')
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
            collate_fn=collate_fn_p1,generator=torch.Generator(device='cuda')

        )
        test_loader = DataLoader(
            test_dataset, batch_size=test_size, shuffle=True,
            collate_fn=collate_fn_p1,generator=torch.Generator(device='cuda')
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


class DKT_Model(Module):
    '''
        Args:
            num_q: the total number of the questions(KCs) in the given dataset
            emb_size: the dimension of the embedding vectors in this model
            hidden_size: the dimension of the hidden vectors in this model
    '''
    def __init__(self, num_q, num_a, emb_size, hidden_size):
        super().__init__()
        self.num_q = num_q
        self.num_a = num_a
        self.emb_size = emb_size
        self.hidden_size = hidden_size

        self.interaction_emb = Embedding(self.num_q * 2, self.emb_size)
        self.lstm_layer = LSTM(
            self.emb_size+num_a, self.hidden_size, batch_first=True
        )
        self.c_integration = CatEmbedding(num_a, emb_size)
        self.out_layer = Linear(self.hidden_size+num_a, self.num_q)
        self.dropout_layer = Dropout()





    def forward(self, q, r, a):
        '''
            Args:
                q: the question(KC) sequence with the size of [batch_size, n]
                r: the response sequence with the size of [batch_size, n]

            Returns:
                y: the knowledge level about the all questions(KCs)
        '''
        x = q + self.num_q * r

        x_emb = self.interaction_emb(x)
        theta_in = self.c_integration(x_emb, a.long())
        h, _ = self.lstm_layer(theta_in)
        theta_out = self.c_integration(h, a.long())
        # print(h.shape)
        theta_out = self.out_layer(theta_out)
        y = self.dropout_layer(theta_out)
        y = torch.sigmoid(y)
        return y


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

        for i in range(1, num_epochs + 1):
            loss_mean = []

            for data in train_loader:
                q, r, qshft, rshft, a, ashft, m = data

                self.train()

                y = self(q.long(), r.long(), a.long())
                # print(y.shape)
                y = (y * one_hot(qshft.long(), self.num_q)).sum(-1)
                # print(y.shape)
                y = torch.masked_select(y, m)
                t = torch.masked_select(rshft, m)

                opt.zero_grad()
                loss = binary_cross_entropy(y, t)
                loss.backward()
                opt.step()

                loss_mean.append(loss.detach().cpu().numpy())

            with torch.no_grad():
                for data in test_loader:
                    q, r, qshft, rshft, a, ashft, m = data

                    self.eval()

                    y = self(q.long(), r.long(), a.long())
                    y = (y * one_hot(qshft.long(), self.num_q)).sum(-1)

                    y = torch.masked_select(y, m).detach().cpu()
                    t = torch.masked_select(rshft, m).detach().cpu()

                    auc = metrics.roc_auc_score(
                        y_true=t.numpy(), y_score=y.numpy()
                    )

                    loss_mean = np.mean(loss_mean)

                    print(
                        "Epoch: {},   AUC: {},   Loss Mean: {}"
                        .format(i, auc, loss_mean)
                    )

                    if auc > max_auc:
                        torch.save(
                            self.state_dict(),
                            os.path.join(
                                ckpt_path, "model.ckpt"
                            )
                        )
                        max_auc = auc

                    aucs.append(auc)
                    loss_means.append(loss_mean)

        return aucs, loss_means




class CatEmbedding(Module):
    def __init__(self, num_a, emb_size) -> None:
        super().__init__()
        self.a_eye = torch.eye(num_a)

        total = num_a
        self.cemb = Linear(num_a, emb_size, bias=False)

    def forward(self, y_e, a):
        # a_e = self.a_eye[a].to(device)
        # ct = torch.cat((a_e), -1) # bz * seq_len * num_fea
        ct = self.a_eye[a].to(device)
        # print(f"ct: {ct.shape}, self.cemb.weight: {self.cemb.weight.shape}")
        # element-wise mul
        Cct = self.cemb(ct) # bz * seq_len * emb
        # print(f"ct: {ct.shape}, Cct: {Cct.shape}")
        theta = torch.mul(y_e, Cct)
        theta = torch.cat((theta, ct), -1)
        return theta
