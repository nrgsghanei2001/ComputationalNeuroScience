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
from learning import STDP, KWTA
from homeostatis import *

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


def raster_plot(spikes, fsize=(5,5)):
    
    plt.figure(figsize=fsize)
    # plt.xlim(0, len(spikes))
    s_spikes = torch.nonzero(spikes)
    plt.scatter(s_spikes[:,0], s_spikes[:,1], s=2, c='darkviolet')
    plt.yticks(np.arange(0, 18))
        
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

def fill_pattern2(pattern, n_free, nstart):
    pattern = pattern.transpose(-2, 1)
    shape = pattern.shape
    new_pattern = torch.zeros(shape[0]+n_free, shape[1])
    for i in range(pattern.shape[0]):
        for j in range(pattern.shape[1]):
            new_pattern[i+nstart][j] = pattern[i][j]

    return new_pattern


def noisy_pattern(pattern1, n=300):
    indexesi = [random.randint(0, 99) for _ in range(n)]
    indexesj = [random.randint(0, 8) for _ in range(n)]

    k = 0
    for i in range(n):
        if pattern1[indexesi[k]][indexesj[k]]:
            pattern1[indexesi[k]][indexesj[k]] = False
        else:
            pattern1[indexesi[k]][indexesj[k]] = True
        k += 1

    return pattern1
            

def neg_pat(pattern):
    new_pat = torch.zeros(pattern.shape[0], pattern.shape[1])
    for i in range(pattern.shape[0]):
        for j in range(pattern.shape[1]):
            if pattern[i][j]:
                new_pat[i][j] = 0
            else:
                new_pat[i][j] = 1

    return new_pat


img1 = image_to_vec('images/barb.tif', (3,3))
# show_image(img1)
img2 = image_to_vec('images/circles.tif', (3,3))
# show_image(img2)
# img3 = image_to_vec('images/lena1.tif', (3,3))
# # show_image(img3)
# img4 = image_to_vec('images/library.tif',(3,3))
# # show_image(img4)
# img5 = image_to_vec('images/Z.jpeg',(3,3))
# show_image(img5)

pe1 = PoissonEncoding(img1, 100, 100)
pe2 = PoissonEncoding(img2, 100, 100)
# pe3 = PoissonEncoding(img3, 280, 100)
# pe4 = PoissonEncoding(img4, 280, 100)
# pe5 = PoissonEncoding(img5, 280, 100)

pe_spikes1 = pe1.encode()
pe_spikes2 = pe2.encode()
# pe_spikes3 = pe3.encode()
# pe_spikes4 = pe4.encode()
# pe_spikes5 = pe5.encode()
# pats = [pe_spikes1, pe_spikes2, pe_spikes3, pe_spikes4, pe_spikes5]
# raster_plot(pe_spikes1)
# raster_plot(pe_spikes2)
# raster_plot(pe_spikes3)
# raster_plot(pe_spikes4)
# raster_plot(pe_spikes5)
# pat = torch.zeros((280, 45))
# start_index = 0
# space_duration = 20

# for pattern in pats:
#     end_index = start_index + 40
#     # Place the pattern in the appropriate section of the output array
#     pat[start_index:end_index, :pattern.shape[1]] = pattern
#     # Update the starting index for the next pattern
#     start_index = end_index + space_duration
pattern1 = fill_pattern2(pe_spikes1, 9, 0)
pattern2 = fill_pattern2(pe_spikes2, 9, 9)
# pattern3 = fill_pattern2(pe_spikes3, 45, 20)
# pattern4 = fill_pattern2(pe_spikes4, 45, 30)
# pattern5 = fill_pattern2(pe_spikes5, 45, 40)
# print(pattern5.shape)
# pattern2 = neg_pat(pe_spikes1)
pattern1 = pattern1.transpose(-2, 1)
pattern2 = pattern2.transpose(-2, 1)
# pattern3 = pattern3.transpose(-2, 1)
# pattern4 = pattern4.transpose(-2, 1)
# pattern5 = pattern5.transpose(-2, 1)
# # pattern1 = fill_pattern(pattern1, 100)
# pattern2 = fill_pattern(pattern2, 0, False)
# pattern1 = pattern1.transpose(-2, 1)
# pattern2 = pattern2.transpose(-2, 1)
raster_plot(pattern1)
raster_plot(pattern2)
# raster_plot(pattern3)
# raster_plot(pattern4)
# raster_plot(pattern5)
# p1 = pattern1.copy()
# p2 = pattern2.copy()
# pattern1 = noisy_pattern(pattern1)
# pattern2 = noisy_pattern(pattern2)
# raster_plot(pat)
# raster_plot(pattern2)


patt = torch.zeros(100, 18)
for i in range(100):
    for j in range(pattern2.shape[1]):
        if i < 40:
            patt[i][j] = pattern1[i][j]
        elif i >= 40 and i < 60:
            patt[i][j] = 0
        elif i >= 60 and i < 100:
            patt[i][j] =pattern2[i-60][j]
        # elif i >= 100 and i < 120:
        #     patt[i][j] = 0
        # elif i >= 120 and i < 160:
        #     patt[i][j] = pattern3[i-120][j]
        # elif i >= 160 and i < 180:
        #     patt[i][j] = 0
        # elif i >= 180 and i < 220:
        #     patt[i][j] = pattern4[i-180][j]
        # elif i >= 220 and i < 240:
        #     patt[i][j] = 0
        # elif i >= 240 and i < 280:
        #     patt[i][j] = pattern5[i-240][j]

raster_plot(patt, fsize=(10, 5))



net = Network(behavior={1: TimeResolution()}, dtype=torch.float64)
input_ng = NeuronGroup(
    9,
    net=net,
    behavior={
        7: InputPattern(pattern=pattern1, pattern2=pattern2, cpt=40, sleep=20, tau_t=10, num_rep=100),
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
            tau=10,
            u_rest=-65,
            u_reset=-73.42,
            threshold=-55,
            R=20,
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
                    4: STDP(lr=[0.3,2],wd=1000, wmin=0, wmax=1),
                    5: Connections(type="full"),    
                    11: Recorder(variables=["W"], tag="layers weights"),  
                })
net.initialize()
net.simulate_iterations(100)

vars1 = []
vars2 = []
time = [i for i in range(100)]
for i in range(9):
    x = []
    y = []
    for j in range(100):
        x.append(net["W", 0][j][i][0])
        y.append(net["W", 0][j][i][1])
    vars1.append(x)
    vars2.append(y)

print(len(vars1[0]))
plt.figure(figsize=(10, 5))
for i in range(9):
    
    # plt.plot(time, vars1[i], label=f"w{i+1}")
    plt.plot(time, vars2[i], label=f"w{i+1}")

plt.xlabel("time")
plt.ylabel("weight")
plt.legend()
plt.show()
plt.plot(net["v", 0])
# plt.plot(net["v", 1])
plt.xlabel("time")
plt.ylabel("v")
plt.legend()
plt.show()
plt.scatter(net["spike", 0][:,0], net["spike", 0][:,1])
# plt.scatter(net["spike", 0][:,1], net["spike", 0][:,1])
plt.ylabel("spike")
plt.xlabel("time")
plt.title("spikes")
plt.legend()
plt.show()


net = Network(behavior={1: TimeResolution()}, dtype=torch.float64)
input_ng = NeuronGroup(
    18,
    net=net,
    behavior={
        7: InputPattern(pattern=pattern1, pattern2=pattern2,cpt=40, sleep=20, tau_t=10),
        8: KWTA(k=1),
        9: ActivityBaseHomeostasis(activity_rate=10, window_size=100, updating_rate=0.8),
        10: EventRecorder("spike", tag="inh_ng1_evrec"),
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
            tau=10,
            u_rest=-65,
            v_reset=-73.42,
            threshold=-55,
            R=20,
            N=2,
        ),
        # 8: KWTA(k=2),
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
                    4: STDP(lr=[1,8],wd=50, wmin=0, wmax=1),
                    5: Connections(type="full"),    
                    11: Recorder(variables=["W"], tag="layers weights"),  
                })

net.initialize()
net.simulate_iterations(100)

vars1 = []
vars2 = []
time = [i for i in range(100)]
for i in range(9):
    x = []
    y = []
    for j in range(100):
        x.append(net["W", 0][j][i][0])
        y.append(net["W", 0][j][i][1])
    vars1.append(x)
    vars2.append(y)

print(len(vars1[0]))
plt.figure(figsize=(10, 5))
for i in range(9):
    
    plt.plot(time, vars1[i], label=f"w{i+1}")
    # plt.plot(time, vars2[i], label=f"w{i+1}")

plt.legend()
plt.xlabel("time")
plt.ylabel("weight")
plt.show()
plt.plot(net["v", 0])
plt.xlabel("time")
plt.ylabel("v")
# plt.plot(net["v", 1])
plt.show()
# plt.scatter(net["spike", 0][:,0], net["spike", 0][:,1], s=1)
plt.scatter(net["spike", 1][:,0], net["spike", 1][:,1])
# plt.scatter(net["spike", 0][:,1], net["spike", 0][:,1])
plt.ylabel("spike")
plt.xlabel("time")
plt.title("spikes")
plt.show()


net = Network(behavior={1: TimeResolution()}, dtype=torch.float64)
input_ng = NeuronGroup(
    54,
    net=net,
    behavior={
        7: InputPattern(pattern=pattern1, pattern2=pattern2, pattern3=pattern3, pattern4=pattern4, pattern5=pattern5, cpt=40, sleep=20, tau_t=10),
        8: KWTA(k=2),
        9: VoltageBaseHomeostasis(target_voltage=-50, max_ta=-55, min_ta=-70)
    },
    tag="inp_NG1",
)


output_ng = NeuronGroup(
    6,
    net=net,
    behavior={
        2: ConstantCurrent(value=0),
        5: InpSyn(),
        7: LIF(
            tau=10,
            u_rest=-65,
            v_reset=-73.42,
            threshold=-55,
            R=20,
            N=1,
        ),
        # 8: KWTA(k=2),
        # 9: ActivityBaseHomeostasis(activity_rate=2, window_size=10, updating_rate=0.2),
        10: Recorder(variables=["v", "I"], tag="inh_ng1_rec, inh_ng1_recorder"),
        11: EventRecorder("spike", tag="inh_ng1_evrec"),
    },
    tag="out_NG2",
)

connect_inp_out = SynapseGroup(net=net,
                src=input_ng, 
                dst=output_ng, 
                behavior={
                    3: SynFun(),
                    4: STDP(lr=[2,12],wd=1000, wmin=0, wmax=0.8),
                    5: Connections(type="full"),    
                    12: Recorder(variables=["W"], tag="layers weights"),  
                })

net.initialize()
net.simulate_iterations(280)

vars1 = []
vars2 = []
vars3 = []
vars4 = []
vars5 = []
vars6 = []
time = [i for i in range(280)]
for i in range(54):
    x = []
    y = []
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    for j in range(280):
        x.append(net["W", 0][j][i][0])
        y.append(net["W", 0][j][i][1])
        y1.append(net["W", 0][j][i][2])
        y2.append(net["W", 0][j][i][3])
        y3.append(net["W", 0][j][i][4])
        y4.append(net["W", 0][j][i][5])
    vars1.append(x)
    vars2.append(y)
    vars3.append(y1)
    vars4.append(y2)
    vars5.append(y3)
    vars6.append(y4)

# print(len(vars1[0]))
# plt.figure(figsize=(10, 5))
# for i in range(54):
    
# #     # plt.plot(time, vars1[i])
#     plt.plot(time, vars1[i], label=f"w{i+1}")
# # plt.plot(time, vars5[i][1])

# plt.xlabel("time")
# plt.ylabel("weight")
# plt.legend()
# plt.show()
plt.plot(net["v", 0])
# plt.plot(net["v", 1])
plt.xlabel("time")
plt.ylabel("v")
plt.show()
# print(net["spike", 0].shape)
plt.scatter(net["spike", 0][:,0], net["spike", 0][:,1])
# plt.scatter(net["spike", 0][:,1], net["spike", 0][:,1])
plt.ylabel("spike")
plt.xlabel("time")
plt.title("spikes")
plt.show()