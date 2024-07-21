import torch
import numpy as np
import torch.distributions as dist

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
        data  = np.abs(self.data.flatten())
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
     
        

class PoissonEncoding:
    """
    Poisson Encoding.
    Calculate the firing rates based on the input data.
    Then, calculate the firing rates per time step.
    Next, store the spike trains over time.
    """

    def __init__(self, data, time, rate_max=5, range_data=(0, 255)):

        self.time = time
        self.data = data
        self.rate =  rate_max
        self.max_val = range_data[1]


    def encode(self):

        rate_i = self.data * self.rate / self.max_val
        rate_dt = rate_i  / self.time
        encoded_data = torch.zeros(self.time, *self.data.shape)   # generate spike patterns randomely based on firing rate from input stimuli
        prob = torch.rand_like(encoded_data)
        for i in range(self.time):
            encoded_data[i] = torch.less(prob[i], rate_dt)


        spikes = torch.zeros(encoded_data.shape[0], encoded_data.shape[1]*encoded_data.shape[2])  # reshape spike pattern to 2D array
        for i in range(spikes.shape[0]):
            for j in range(encoded_data.shape[1]):
                for k in range(encoded_data.shape[2]):
                    spikes[i][j*encoded_data.shape[2]+k] = encoded_data[i][j][k]

        return spikes
    

class Poisson(torch.nn.Module):
    """
    Poisson encoding.
    Input values should be between 0 and 1.
    The intervals between two spikes are picked using Poisson Distribution.

    Args:
        time_window (int): The interval of the coding.
        ratio (float): A scale factor for probability of spiking.
    """

    def __init__(self, time_window, ratio):
        self.time_window = time_window
        self.ratio = ratio

    def __call__(self, img):
        if type(img) is tuple:
            return tuple([self(sub_inp) for sub_inp in img])

        # https://github.com/BindsNET/bindsnet/blob/master/bindsnet/encoding/encodings.py
        original_shape, original_size = img.shape, img.numel()
        flat_img = img.view((-1,)) * self.ratio
        non_zero_mask = flat_img != 0

        flat_img[non_zero_mask] = 1 / flat_img[non_zero_mask]

        dist = torch.distributions.Poisson(rate=flat_img, validate_args=False)
        intervals = dist.sample(sample_shape=torch.Size([self.time_window]))
        intervals[:, non_zero_mask] += (intervals[:, non_zero_mask] == 0).float()

        times = torch.cumsum(intervals, dim=0).long()
        times[times >= self.time_window + 1] = 0

        spikes = torch.zeros(
            self.time_window + 1, original_size, device=img.device, dtype=torch.bool
        )
        spikes[times, torch.arange(original_size, device=img.device)] = True
        spikes = spikes[1:]

        return spikes.view(self.time_window, *original_shape)