from pymonntorch import Behavior
import torch
import numpy as np
import random


class Connections(Behavior):
    def initialize(self, sg):
        self.connection_type = self.parameter("type", "random_prob")
        self.coupling_prob = self.parameter("c_prob", 0.1)
        self.num_partners = self.parameter("p_partners", 0.1)
        self.def_val = self.parameter("def_val", 50)
        self.size_pre  = sg.W.shape[0]
        self.size_post = sg.W.shape[1]


        if self.connection_type == 'full':
            self.connect_full(sg)
        elif self.connection_type == 'random_prob':
            self.connect_random_probability(sg, self.coupling_prob)
        elif self.connection_type == 'random_num_partners':
            self.connect_random_num_partners(sg, self.num_partners)

    def connect_full(self, sg):
        # establish connections between all pairs of neurons (full connectivity)
        sg.W = sg.matrix(mode=self.def_val)

        # uniform weights

        # s = self.def_val // 2
        # uni = [random.randint(self.def_val-s, self.def_val+s) for _ in range(int(self.size_post*self.size_pre))]
        # k = 0
        # for i in range(self.size_pre):
        #     for j in range(self.size_post):
        #         sg.W[i][j] = torch.tensor(uni[k])
        #         k += 1
               
            

    def connect_random_probability(self, sg, coupling_probability):
        # establish connections with a fixed coupling probability
        # uniform weights
        # s = self.def_val // 2
        # uni = [random.randint(self.def_val-s, self.def_val+s) for _ in range(int(self.size_post*self.size_pre*self.coupling_prob*2))]
        # k = 0
        for i in range(self.size_pre):
            for j in range(self.size_post):
                if np.random.rand() < coupling_probability:
                    sg.W[i][j] = torch.tensor(self.def_val)
                    # sg.W[i][j] = torch.tensor(uni[k])
                    # k += 1
                else:
                    sg.W[i][j] = torch.tensor(0)

    def connect_random_num_partners(self, sg, p_presynaptic_partners):
        # establish connections with a fixed number of presynaptic partners
        num_partners = int(p_presynaptic_partners * self.size_pre)
        # uniform weights
        # s = self.def_val // 2
        # uni = [random.randint(self.def_val-s, self.def_val+s) for _ in range(int(self.size_post*self.size_pre*p_presynaptic_partners))]
        # k = 0
        for j in range(self.size_post):
            presynaptic_indices = np.random.choice(self.size_pre, size=num_partners, replace=False)
            for i in range(self.size_pre):
                if i in presynaptic_indices:
                    # sg.W[i][j] = torch.tensor(uni[k])
                    # k+=1
                    sg.W[i][j] = torch.tensor(self.def_val)
                else:
                    sg.W[i][j] = torch.tensor(0)

    def forward(self, sg):
        pass
