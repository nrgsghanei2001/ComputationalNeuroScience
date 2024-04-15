import torch



class TimeToFirstSpikeEncoding:
    
    def __init__(self, range, data, time, num_neurons):
        self.range0 = range[0]
        self.range1 = range[1]
        self.time = time
        self.num_neurons = num_neurons
        self.data = data

    def scale_data(self):
        self.data = abs(self.data - self.range1 + self.range0)
        # Scaling data to values to time range
        self.data = (self.data - self.range0) * (self.time - 1) // (self.range1 - self.range0)

    def encode(self):
        self.scale_data()
        print(self.data.shape)
        spikes = torch.Tensor(self.time, self.num_neurons)
        for i in range(self.time):
            spikes[i] = self.data == i
        
        return spikes
        # _data = self.data.flatten()
        # spikes = torch.zeros((self.time + 1, self.num_neurons))
        # times = self.time - (_data * (self.time / _data.max())).long()
        # print(times.shape)
        # spikes[times, torch.arange(self.num_neurons)] = 1

        # spikes = spikes[:-1]
        # return spikes