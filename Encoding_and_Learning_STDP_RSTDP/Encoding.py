import torch



class TimeToFirstSpikeEncoding:
    
    def __init__(self, range, data, time, num_neurons):
        self.range0 = range[0]
        self.range1 = range[1]
        self.time = time
        self.num_neurons = num_neurons
        self.data = data

    def scale_data(self):
        _data = self.data.flatten()
        times = self.time - (_data * (self.time / _data.max())).long()
        return times
        # self.data = abs(self.data - self.range1 + self.range0)
        # # Scaling data to values to time range
        # self.data = (self.data - self.range0) * (self.time - 1) // (self.range1 - self.range0)

    def encode(self):
        times = self.scale_data()
        # spikes = torch.zeros((self.time + 1, self.num_neurons))
        # spikes[times, torch.arange(self.num_neurons)] = 1
        # spikes = spikes[:-1]
        # return spikes
    
        spikes = torch.zeros((self.time, self.num_neurons))
        for j in range(self.num_neurons):
            for i in range(self.time):
                if i == times[j]:
                    spikes[i][j] = 1

        
        return spikes
        
        