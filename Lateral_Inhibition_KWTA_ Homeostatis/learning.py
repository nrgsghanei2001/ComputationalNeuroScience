from pymonntorch import Behavior
import torch
import numpy as np


class STDP(Behavior):
    def initialize(self, sg):
        self.lr = self.parameter("lr", [0.1, 0.1])
        self.weight_decay = self.parameter("wd")
        self.wmin = self.parameter("wmin", 0)
        self.wmax = self.parameter("wmax", 1)


    def forward(self, sg):
        # print("==================")
        # print(sg.W)
        # print("==================")
        mask = torch.ones(*sg.W.size())
        mask0 = torch.zeros(*sg.W.size())
        src_s = self.mask_spike_trace(sg.src.spike, mask)  # source spikes
        dst_s = sg.dst.spike * mask                        # destination spikes
        src_t = self.mask_spike_trace(sg.src.trace, mask)  # source trace
        dst_t = sg.dst.trace * mask                        # destination trace
        
        A1 = self.lr[0] * src_s * dst_t
        A2 = self.lr[1] * dst_s * src_t

        # sg.W += (-A1 + A2)
        sg.W += -(sg.W/self.weight_decay) +(-A1 + A2)
        sg.W = np.clip(sg.W, self.wmin, self.wmax)
        
        # print("==================")
        # print(src_s)
        # print(dst_s)
        # # print(dst_t)
        # print(-A1)
        # print(sg.W)
        # print("==================")

        self.apply_lateral_inhibition(sg)

    
    def apply_lateral_inhibition(self, sg):
        
        if sg.dst.spike[0] and (not (sg.dst.spike[1])):
            sg.dst.I[1] -= 15
            # sg.dst.I[2] -= 15
            # sg.dst.I[3] -= 15
            # sg.dst.I[4] -= 15
            # sg.dst.I[5] -= 15
        elif sg.dst.spike[1] and (not (sg.dst.spike[0])):
            sg.dst.I[0] -= 5
            # sg.dst.I[2] -= 15
            # sg.dst.I[3] -= 15
            # sg.dst.I[4] -= 15
            # sg.dst.I[5] -= 15
        # elif sg.dst.spike[2] and (not (sg.dst.spike[0] and sg.dst.spike[1] and sg.dst.spike[3] and sg.dst.spike[4] and sg.dst.spike[5])):
        #     sg.dst.I[0] -= 5
        #     sg.dst.I[1] -= 15
        #     sg.dst.I[3] -= 15
        #     sg.dst.I[4] -= 15
        #     sg.dst.I[5] -= 15
        # elif sg.dst.spike[3] and (not (sg.dst.spike[0] and sg.dst.spike[2] and sg.dst.spike[1] and sg.dst.spike[4] and sg.dst.spike[5])):
        #     sg.dst.I[0] -= 5
        #     sg.dst.I[2] -= 15
        #     sg.dst.I[1] -= 15
        #     sg.dst.I[4] -= 15
        #     sg.dst.I[5] -= 15
        # elif sg.dst.spike[4] and (not (sg.dst.spike[0] and sg.dst.spike[2] and sg.dst.spike[1] and sg.dst.spike[3] and sg.dst.spike[5])):
        #     sg.dst.I[0] -= 5
        #     sg.dst.I[2] -= 15
        #     sg.dst.I[1] -= 15
        #     sg.dst.I[3] -= 15
        #     sg.dst.I[5] -= 15
        # elif sg.dst.spike[5] and (not (sg.dst.spike[0] and sg.dst.spike[2] and sg.dst.spike[1] and sg.dst.spike[3] and sg.dst.spike[4])):
        #     sg.dst.I[0] -= 5
        #     sg.dst.I[2] -= 15
        #     sg.dst.I[1] -= 15
        #     sg.dst.I[3] -= 15
        #     sg.dst.I[4] -= 15


    # def apply_k_winners_take_all(self, sg):

        
        


    def mask_spike_trace(self, inp, mask):
        if len(inp.size()) > 1:
            inp = torch.t(inp)
        else:
            inp = inp.view(inp.size(0), -1)

        return inp * mask



class RSTDP(Behavior):


    def initialize(self, sg):
        self.lr = self.parameter("lr", [10, 15])
        self.weight_decay = self.parameter("wd", 2)
        self.tau_c = self.parameter("tau_c", 1000)
        self.wmin = self.parameter("wmin", 10)
        self.wmax = self.parameter("wmax", 30)
        self.pct = self.parameter("pct", 40)
        self.d = self.parameter("d", 1)
        self.tau_d = self.parameter("tau_d", 10)
        sg.C = torch.zeros(*sg.W.size())
        # self.d_values =  self.parameter("d_val", [0.75, -0.25, -2])



    def forward(self, sg):

        mask = torch.ones(*sg.W.size())
        src_s = self.mask_spike_trace(sg.src.spike, mask)
        dst_s = sg.dst.spike * mask
        src_t = self.mask_spike_trace(sg.src.trace, mask)
        dst_t = sg.dst.trace * mask
        
        
        A1 = self.lr[0] * dst_t * src_s 
        A2 = self.lr[1] * src_t * dst_s 
        stdp = -A1 + A2

        da =  self.reward(sg, sg.dst.spike)
        self.d = -(self.d/self.tau_d) + da
        
        dc_dt = -sg.C/self.tau_c + stdp * ((src_s + dst_s) > 0)
        sg.C +=  dc_dt

        sg.W += sg.C * self.d

        sg.W = np.clip(sg.W, self.wmin, self.wmax)



    def mask_spike_trace(self, inp, mask):
        if len(inp.size()) > 1:
            inp = torch.t(inp)
        else:
            inp = inp.view(inp.size(0), -1)

        return inp * mask
    
    def reward(self, sg, output):
        

        input_pattern = sg.src.pn 
        print(input_pattern, output)
        if output[0] == False and output[1] == False:
            return 0
        elif output[0] == True and output[1] == True:
            return -1
        elif input_pattern == 0 and output[0] == True:
            return 2
        # elif input_pattern == 0 and output[1] == True:
        #     return 3
        elif input_pattern == 1 and output[1] == True:
            return 2
        # elif input_pattern == 1 and output[0] == True:
        #     return 3
        else:
            return -1
        




class KWTA(Behavior):


    def __init__(self, k, *args, dimension=None, **kwargs):
        super().__init__(*args, k=k, dimension=dimension, **kwargs)

    def initialize(self, neurons):
        self.k = self.parameter("k", None, required=True)
        self.dimension = self.parameter("dimension", None)
        self.shape = (neurons.size, 1, 1)
        # if hasattr(neurons, "depth"):
        #     self.shape = (neurons.depth, neurons.height, neurons.width)

    def forward(self, neurons):
        will_spike = neurons.v >= neurons.threshold
        v_values = neurons.v

        dim = 0
        if self.dimension is not None:
            v_values = v_values.view(self.shape)
            will_spike = will_spike.view(self.shape)
            dim = self.dimension

        if (will_spike.sum(axis=dim) <= self.k).all():
            return

        _, k_winners_indices = torch.topk(
            v_values, self.k, dim=dim, sorted=False
        )

        ignored = will_spike
        ignored.scatter_(dim, k_winners_indices, False)

        neurons.v[ignored.view((-1,))] = neurons.v_reset       