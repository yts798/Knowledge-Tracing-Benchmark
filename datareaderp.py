import os

import pickle
import json
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

from utils import match_seq_len_p3



class readerp3(Dataset):
    def __init__(self, seq_len, dataset_name) -> None:
        super().__init__()

        self.seq_len = seq_len

        with open("config_data.json") as f:
            config = json.load(f)
            data_config = config[dataset_name]

        self.dataset_dir = data_config['directory']
        self.dataset_path = os.path.join(
            self.dataset_dir, data_config['filename']
        )

        self.kc = data_config['kc']
        self.outcome = data_config['outcome']
        self.student = data_config['student']
        self.a = data_config['a']
        self.b = data_config['b']
        self.c = data_config['c']
        print("I AM USING ", data_config['a'], "and", data_config['b'], "and", data_config['c'])

        # self.q_seqs, self.r_seqs, self.q_list, self.u_list, self.q2idx, self.u2idx, self.a_seqs, self.a_list, self.a2idx, self.b_seqs, self.b_list, self.b2idx = self.preprocess()
        # if False:
        if os.path.exists(os.path.join(self.dataset_dir, "q_seqs.pkl")):
            print("Found existing pkl")
            with open(os.path.join(self.dataset_dir, "q_seqs.pkl"), "rb") as f:
                self.q_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "r_seqs.pkl"), "rb") as f:
                self.r_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "q_list.pkl"), "rb") as f:
                self.q_list = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "u_list.pkl"), "rb") as f:
                self.u_list = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "q2idx.pkl"), "rb") as f:
                self.q2idx = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "u2idx.pkl"), "rb") as f:
                self.u2idx = pickle.load(f)

            with open(os.path.join(self.dataset_dir, "a_seqs.pkl"), "rb") as f:
                self.a_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "a_list.pkl"), "rb") as f:
                self.a_list = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "a2idx.pkl"), "rb") as f:
                self.a2idx = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "b_seqs.pkl"), "rb") as f:
                self.b_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "b_list.pkl"), "rb") as f:
                self.b_list = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "b2idx.pkl"), "rb") as f:
                self.b2idx = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "c_seqs.pkl"), "rb") as f:
                self.c_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "c_list.pkl"), "rb") as f:
                self.c_list = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "c2idx.pkl"), "rb") as f:
                self.c2idx = pickle.load(f)


        else:
            self.q_seqs, self.r_seqs, self.q_list, self.u_list, self.q2idx, \
            self.u2idx, self.a_seqs, self.a_list, self.a2idx, self.b_seqs, self.b_list, self.b2idx, self.c_seqs, self.c_list, self.c2idx = self.preprocess()

        self.num_u = self.u_list.shape[0]
        self.num_q = self.q_list.shape[0]
        self.num_a = self.a_list.shape[0]
        self.num_b = self.b_list.shape[0]
        self.num_c = self.c_list.shape[0]
        if self.seq_len:
            self.q_seqs, self.r_seqs, self.a_seqs, self.b_seqs, self.c_seqs = \
                match_seq_len_p3(self.q_seqs, self.r_seqs, self.a_seqs, self.b_seqs, self.c_seqs, self.seq_len)

        self.len = len(self.q_seqs)

    # def __getitem__(self, index):
    #     return self.q_seqs[index], self.r_seqs[index],

    def __getitem__(self, index):

        return self.q_seqs[index], self.r_seqs[index], self.a_seqs[index], self.b_seqs[index], self.c_seqs[index]

    def __len__(self):
        return self.len

    def preprocess(self):
        df = pd.read_csv(self.dataset_path, low_memory=False, nrows = 1000000) \
            .dropna(subset=[self.kc, self.a, self.b, self.c])
        # sorted is omitted

        u_list = np.unique(df[self.student].values)
        q_list = np.unique(df[self.kc].values)

        a_list = np.unique(df[self.a].values)
        b_list = np.unique(df[self.b].values)
        c_list = np.unique(df[self.c].values)

        u2idx = {u: idx for idx, u in enumerate(u_list)}
        q2idx = {q: idx for idx, q in enumerate(q_list)}

        a2idx = {a: idx for idx, a in enumerate(a_list)}
        b2idx = {b: idx for idx, b in enumerate(b_list)}
        c2idx = {c: idx for idx, c in enumerate(c_list)}
        q_seqs = []
        r_seqs = []

        a_seqs = []
        b_seqs = []
        c_seqs = []
        for u in u_list:
            u_df = df[df[self.student] == u]

            q_seqs.append([q2idx[q] for q in u_df[self.kc].values])
            r_seqs.append(u_df[self.outcome].values)

            a_seqs.append([a2idx[a] for a in u_df[self.a].values])
            b_seqs.append([b2idx[b] for b in u_df[self.b].values])
            c_seqs.append([c2idx[c] for c in u_df[self.c].values])
        with open(os.path.join(self.dataset_dir, "q_seqs.pkl"), "wb") as f:
            pickle.dump(q_seqs, f)
        with open(os.path.join(self.dataset_dir, "r_seqs.pkl"), "wb") as f:
            pickle.dump(r_seqs, f)
        with open(os.path.join(self.dataset_dir, "q_list.pkl"), "wb") as f:
            pickle.dump(q_list, f)
        with open(os.path.join(self.dataset_dir, "u_list.pkl"), "wb") as f:
            pickle.dump(u_list, f)
        with open(os.path.join(self.dataset_dir, "q2idx.pkl"), "wb") as f:
            pickle.dump(q2idx, f)
        with open(os.path.join(self.dataset_dir, "u2idx.pkl"), "wb") as f:
            pickle.dump(u2idx, f)

        with open(os.path.join(self.dataset_dir, "a_seqs.pkl"), "wb") as f:
            pickle.dump(a_seqs, f)
        with open(os.path.join(self.dataset_dir, "a_list.pkl"), "wb") as f:
            pickle.dump(a_list, f)
        with open(os.path.join(self.dataset_dir, "a2idx.pkl"), "wb") as f:
            pickle.dump(a2idx, f)
        with open(os.path.join(self.dataset_dir, "c_seqs.pkl"), "wb") as f:
            pickle.dump(c_seqs, f)
        with open(os.path.join(self.dataset_dir, "c_list.pkl"), "wb") as f:
            pickle.dump(c_list, f)
        with open(os.path.join(self.dataset_dir, "c2idx.pkl"), "wb") as f:
            pickle.dump(c2idx, f)


        return q_seqs, r_seqs, q_list, u_list, q2idx, u2idx, a_seqs, a_list, a2idx, b_seqs, b_list, b2idx, c_seqs, c_list, c2idx
