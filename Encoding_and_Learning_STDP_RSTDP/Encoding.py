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
        self.std = 1

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
            print(self.gaussian_distribution(i.item()))

        
        return spikes
        
        