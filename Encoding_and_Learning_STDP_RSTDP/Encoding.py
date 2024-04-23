import torch
import numpy as np


class TimeToFirstSpikeEncoding:
    
    def __init__(self, data, time, num_neurons):
        self.time = time
        self.num_neurons = num_neurons
        self.data = data

    def scale_data(self):
        _data = self.data.flatten()
        times = self.time - (_data * (self.time / _data.max())).long()
        return times

    def encode(self):
        times = self.scale_data()
        spikes = torch.zeros((self.time, self.num_neurons))
        for j in range(self.num_neurons):
            for i in range(self.time):
                if i == times[j]:
                    spikes[i][j] = 1

        
        return spikes



class GaussianEncoding:

    def __init__(self, time, num_neurons, data):
        self.time = time
        self.num_neurons = num_neurons
        self.data = data
        self.std = 0.5

    def scale_data(self):
        _data = self.data.flatten()
        times = self.num_neurons - (_data * (self.num_neurons / _data.max())).long()
        return times
    
    def gaussian_distribution(self, x):
        gaussian = []
        print(x)
        for mean in range(self.num_neurons):
            gaussian.append((1/(self.std*np.sqrt(2*np.pi))) * np.e ** ((-1/2)*((x-mean/self.std )** 2)))
        return(gaussian)
    
    def encode(self):
        times = self.scale_data()
        spikes = torch.zeros((self.time, self.num_neurons))
        # for j in range(self.num_neurons):
        #     for i in range(self.time):
        #         if i == times[j]:
        #             spikes[i][j] = 1
        for i in times:
            gd = self.gaussian_distribution(i.item())
            for j, x in enumerate(gd):
                if x >= 0.01:
                    print(x, end=" ")
            print()

        
        return spikes


class PositionEncoder:
    """
    Position coding.
    """

    def __init__(self, data, time, node_n, range_data=(0, 255), std=0.5):
        self.data = data
        self.time = time
        self.node_n = node_n
        self.range_data = range_data
        self.std = std

        # Making gaussian functions represnting our nodes
        self.functions = [self.gaussianFunc(i/self.node_n, self.std) for i in range(self.node_n)]
        
    def gaussianFunc(self, mean, std) -> callable:
        def f(x):
            return (1/(std*np.sqrt(2*np.pi))) * np.e ** ((-1/2)*((x-mean/std )** 2))
        return f

    def encode(self):
        self.data = self.data.flatten()
        shape = (self.data.shape)[0]
        self.data = self.data.long() / self.range_data[1]
        
        # spikes = torch.zeros(self.node_n, self.time)
        times = torch.zeros(self.node_n, shape)
        times2 = torch.zeros(shape, self.node_n)
        k = 0
        for i in range(len(self.functions)):
            data_ = self.data.clone()        
            data_.apply_(self.functions[i])
            data_ = abs(data_ - 1/(self.std*np.sqrt(2*np.pi)))
            for j in range(shape):
                times[k][j] = data_[j]  
            k += 1
        maxx = 0
        for i in range(times2.shape[0]):
            for j in range(times2.shape[1]):
                times2[i][j] = times[j][i]
                if times2[i][j] > maxx:
                    maxx = times2[i][j]
        print(maxx)
        times3 = torch.zeros(shape, self.node_n)
        for i in range(times2.shape[0]):
            for j in range(times2.shape[1]):
                if times2[i][j] > 0.1:
                    x = (times2[i][j] - 0.1) / (maxx - 0.1)
                    x = 1 - x
                    times3[i][j] = int(x * self.time)
                else:
                    times3[i][j] = -1
        print(times3)
        spikes = torch.zeros(self.time, self.node_n * shape)
        for i in range(times2.shape[0]):
            for j in range(times2.shape[1]):
                if times3[i][j] != -1:
                    spikes[int(times3[i][j].item())][i*self.node_n + j] = 1
        return spikes
     
        

class PoissonEncoder:
    """
    Poisson coding.

    Implement Poisson coding.
    """

    def __init__(self, data, time, rate_max=None, data_max_val= 255):
        self.time = time
        self.data = data

        if rate_max is None:
            self.rate_max =  time
        else:
            self.rate_max = rate_max
        self.data_max_val = data_max_val
        self.r = 0.06

    def encode(self):


        r_x = self.data * self.rate_max / self.data_max_val
        r_x_dt = r_x  / self.time
        encoded_data = torch.zeros(self.time, *self.data.shape)
        p = torch.rand_like(encoded_data)
        for t in range(self.time):
            encoded_data[t] = torch.less(p[t], r_x_dt)

        spikes = torch.zeros(encoded_data.shape[0], encoded_data.shape[1]*encoded_data.shape[2])
        for i in range(spikes.shape[0]):
            for j in range(encoded_data.shape[1]):
                for k in range(encoded_data.shape[2]):
                    spikes[i][j*encoded_data.shape[2]+k] = encoded_data[i][j][k]
        return spikes