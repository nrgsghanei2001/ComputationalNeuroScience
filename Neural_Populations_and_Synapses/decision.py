from pymonntorch import Behavior
import torch
import numpy as np


class Decision(Behavior):
    def initialize(self, ng_exc1, ng_exc2, ng_inh):
        self.Wee = self.parameter("Wee", 0.1)
        self.Wei = self.parameter("Wei", -0.2)
        self.tau_e = self.parameter("tau_e", 10)
        self.R = self.parameter("R", 1)
        ng_exc1.he1 = 0
        ng_exc2.he2 = 0
        ng_inh.hinh = 0

    def forward(self, ng_exc1, ng_exc2, ng_inh):
        he1 = ng_exc1.I
        he2 = ng_exc2.I
        hinh = ng_inh.I

        he1 += ((-he1) + self.Wee * ng_exc1.population_activity + self.Wei * ng_inh.population_activity + self.R * ng_exc1.inputI) / self.tau_e
        he2 += ((-he2) + self.Wee * ng_exc2.population_activity + self.Wei * ng_inh.population_activity + self.R * ng_exc2.inputI) / self.tau_e
        hinh += ((-hinh) + self.Wei * ng_exc1.population_activity + self.Wei * ng_exc2.population_activity) / self.tau_e
