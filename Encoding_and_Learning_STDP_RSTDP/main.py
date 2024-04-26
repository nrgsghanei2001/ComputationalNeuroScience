import cv2
import torch
from Encoders import TimeToFirstSpikeEncoding, GaussianEncoding, PoissonEncoding
from matplotlib import pyplot as plt
from pymonntorch import *
import numpy as np
from timeRes import TimeResolution
from current import *
from lif import LIF, InputPattern
from synapse import *
from connections import Connections
from learning import *

# read an image and convert it to torch tensor
def image_to_vec(address, size=(10, 10)):
    img = cv2.imread(address)
    img = cv2.resize(img, (size[0], size[1]))
    img = torch.from_numpy(img)
    img = img.sum(2)//3

    return img


def show_image(img):
    plt.imshow(img, cmap='gray')  
    plt.show()

def raster_plot(spikes):
    
    plt.figure(figsize=(5,5))
    plt.xlim(0, len(spikes))
    s_spikes = torch.nonzero(spikes)
    plt.scatter(s_spikes[:,0], s_spikes[:,1], s=2, c='darkviolet')
    
        
    plt.xlabel("Time")
    plt.ylabel("Neurons")
    plt.show()

def fill_pattern(pattern, n_free, is_right=True):
    pattern = pattern.transpose(-2, 1)
    shape = pattern.shape
    new_pattern = torch.zeros(shape[0]+n_free, shape[1])
    if is_right:
        for i in range(pattern.shape[0]):
            for j in range(pattern.shape[1]):
                new_pattern[i][j] = pattern[i][j] 
    else:
        for i in range(pattern.shape[0]):
            for j in range(pattern.shape[1]):
                new_pattern[i+n_free][j] = pattern[i][j]

    return new_pattern


img1 = image_to_vec('images/barb.tif', (3, 3))
img2 = image_to_vec('images/circles.tif', (3, 3))

pe1 = PoissonEncoding(img1, 100, 200)
pe2 = PoissonEncoding(img2, 100, 200)

pe_spikes1 = pe1.encode()
pe_spikes2 = pe2.encode()

raster_plot(pe_spikes1)
raster_plot(pe_spikes2)

pattern1 = fill_pattern(pe_spikes1, 9)
pattern2 = fill_pattern(pe_spikes2, 9, False)
raster_plot(pattern1)
raster_plot(pattern2)
pattern1 = pattern1.transpose(-2, 1)
pattern2 = pattern2.transpose(-2, 1)



net = Network(behavior={1: TimeResolution()}, dtype=torch.float64)
input_ng = NeuronGroup(
    18,
    net=net,
    behavior={
        7: InputPattern(pattern=pattern1, pattern2=pattern2, cpt=40, sleep=20),
    },
    tag="inp_NG1",
)


output_ng = NeuronGroup(
    2,
    net=net,
    behavior={
        2: ConstantCurrent(value=0),
        5: InpSyn(),
        7: LIF(
            tau=20,
            u_rest=-65,
            u_reset=-73.42,
            threshold=-35,
            R=5,
            N=2,
        ),
        9: Recorder(variables=["v", "I"], tag="inh_ng1_rec, inh_ng1_recorder"),
        10: EventRecorder("spike", tag="inh_ng1_evrec"),
    },
    tag="out_NG2",
)

connect_inp_out = SynapseGroup(net=net,
                src=input_ng, 
                dst=output_ng, 
                behavior={
                    3: SynFun(),
                    4: STDP(),
                    5: Connections(def_val=30, type="full"),    
                    11: Recorder(variables=["W"], tag="layers weights"),  
                })

net.initialize()
net.simulate_iterations(100)

net1 = net["W", 0][-1]
column1 = []  # First column
column2 = []  # First column
for i in range(net1.shape[0]):
    column1.append(net1[i][0].item())
    column2.append(net1[i][1].item())
dot_product = np.dot(column1, column2)
magnitude1 = np.linalg.norm(column1)
magnitude2 = np.linalg.norm(column2)
cosine_similarity = dot_product / (magnitude1 * magnitude2)
print("cosine: ",cosine_similarity)


vars1 = []
vars2 = []
time = [i for i in range(100)]
for i in range(18):
    x = []
    y = []
    for j in range(100):
        x.append(net["W", 0][j][i][0])
        y.append(net["W", 0][j][i][1])
    vars1.append(x)
    vars2.append(y)

for i in range(18):
    
    plt.plot(time, vars1[i])
    plt.plot(time, vars2[i])
