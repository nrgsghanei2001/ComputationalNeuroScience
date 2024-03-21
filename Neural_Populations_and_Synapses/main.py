from pymonntorch import Network, NeuronGroup, Recorder, EventRecorder, SynapseGroup
from lif import LIF
from timeRes import TimeResolution
from synapse import SynFun, InpSyn
from current import *
import torch
from matplotlib import pyplot as plt


net = Network(behavior={1: TimeResolution()}, dtype=torch.float64)
ng1 = NeuronGroup(
    1,
    net=net,
    behavior={
        2: ConstantCurrent(value=50),
        4: InpSyn(),
        5: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-73.42,
            threshold=-13,
            R=1,
        ),
        9: Recorder(variables=["v", "I"], tag="ng1_rec, ng1_recorder"),
        10: EventRecorder("spikes", tag="ng1_evrec"),
    },
    tag="NG1",
)

SynapseGroup(net=net,
            src=ng1, 
            dst=ng1, 
            behavior={
                 3: SynFun(),
            })

net.initialize()
net.simulate_iterations(100)


plt.plot(net["v", 0][:,:20])
plt.show()
plt.plot(net["I", 0][:,:20])
plt.show()

plt.scatter(net["spike", 0][:,0], net["spike", 0][:,1])
plt.show()
