import torch
import numpy as np


class TimeToFirstSpikeEncoding:
    """
    Time-to-First-Spike encoding
    Each neuron represents a pixel and the value of it, 
    is the time that neuron spikes.
    """
     
    def __init__(self, data, time):
        self.time        = time    # simulation time
        self.data        = data  # input stimuli (image)
        self.num_neurons = self.data.shape[0] * self.data.shape[1]   # number of neurons

    def scale_data(self):     # map pixels values to interval of simulation time
        data  = self.data.flatten()
        times = self.time - (data * (self.time / data.max())).long()
        return times

    def encode(self):
        times  = self.scale_data()
        spikes = torch.zeros((self.time, self.num_neurons))
        for neuron_index in range(self.num_neurons):
            for time_index in range(self.time):
                if time_index == times[neuron_index]:
                    spikes[time_index, neuron_index] = 1   # build spike pattern
 
        return spikes



class GaussianEncoding:
    """
    Gaussian Encoding.
    Each pixel has n neurons representing a normal distribution with mean of i.
    The neuron that has higher value for given pixel, spikes sooner.
    Values below a threshold are being ignored.
    """

    def __init__(self, data, time, num_nodes, range_data=(0, 255), std=0.5):
        self.data       = data
        self.time       = time
        self.num_nodes  = num_nodes
        self.range_data = range_data
        self.std        = std
        self.gaussian   = [self.calc_gaussian(i/self.num_nodes, self.std) for i in range(self.num_nodes)] # gaussian distribution for nodes
        

    def calc_gaussian(self, mean, std) -> callable:
        def f(x):
            return (1/(std*np.sqrt(2*np.pi))) * (np.e ** ((-1/2)*((x-mean/std )** 2)))
        return f


    def scale_data(self):
        self.data = self.data.flatten()
        self.data = self.data.long() / self.range_data[1]


    def encode(self):
        self.scale_data()
        shape = (self.data.shape)[0]

        times = torch.zeros(self.num_nodes, shape)
        for k, calc_gaussian in enumerate(self.gaussian):   # calculate each gaussian value for each pixel
            modified_data = self.data.clone().apply_(calc_gaussian)
            difference = torch.abs(modified_data - 1 / (self.std * np.sqrt(2 * np.pi)))
            times[k] = difference

        max_value = 0
        times_T = torch.zeros(shape, self.num_nodes)    # transpose the values
        for i in range(times_T.shape[0]):
            for j in range(times_T.shape[1]):
                times_T[i, j] = times[j, i]
                max_value = max(max_value, times_T[i, j])

        threshold = 0.1
        times = torch.zeros(shape, self.num_nodes)
        for i in range(times_T.shape[0]):                  # find the time that each neuron is being active due to threshold and interval
            for j in range(times_T.shape[1]):
                if times_T[i, j] > threshold:
                    x = (times_T[i, j] - threshold) / (max_value - threshold)
                    x = 1 - x
                    times[i, j] = int(x * self.time)
                else:
                    times[i, j] = -1

        spikes = torch.zeros(self.time, self.num_nodes * shape)       # build spike pattern
        for i in range(times_T.shape[0]):
            for j in range(times_T.shape[1]):
                if times[i][j] != -1:
                    spikes[int(times[i][j].item())][i*self.num_nodes + j] = 1
                    
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