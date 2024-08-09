import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import cv2
from pymonntorch import NeuronGroup, SynapseGroup, NeuronDimension, EventRecorder, Recorder
from conex import (
    Neocortex,
    InputLayer,
)
from conex.behaviors.neurons import (
    SimpleDendriteStructure,
    SimpleDendriteComputation,
    LIF,
    SpikeTrace,
    NeuronAxon,
)
from conex.behaviors.synapses import (
    SynapseInit,
    WeightInitializer,
    Conv2dDendriticInput,
    Conv2dSTDP,
)
from conex.behaviors.neurons.specs import KWTA
from conex.behaviors.neurons.homeostasis import ActivityBaseHomeostasis
from conex.helpers.transforms.misc import DivideSignPolarity, SqueezeTransform
from conex.helpers.transforms.encoders import Poisson
from torch.utils.data import DataLoader
from visualization import visualize_network_structure
from filters import DoGFilter
from dendrites import MaxPool2D
from transforms import Conv2dFilter

##################################################
# parameters
##################################################
DEVICE = "cpu"
DTYPE = torch.float32
DT = 1
POISSON_TIME = 5
POISSON_RATIO = 5e-2
DATA_ROOT = "C:/Users/ASC/OneDrive/Desktop/temp/my_img/"
SENSORY_SIZE_HEIGHT = 28
SENSORY_SIZE_WIDTH = 28
SENSORY_TRACE_TAU_S = 2.7

# DoG filter parameters
DOG_FILTER_SIZE = (1, 1, 15, 15)
DOG_FILTER_SIGMA1 = 1.0
DOG_FILTER_SIGMA2 = 5.0

# Maxpooling parameters
MAXPOOL_KERNEL_SIZE = 5
MAXPOOL_STRIDE = 1

# Feature extraction layer
FEATURE_MAPS = 1
FEATURE_MAP_SIZE = (FEATURE_MAPS, 1, 50, 50)
K_WINNERS = 4
L4_EXC_DEPTH = 4
L4_EXC_HEIGHT = 24
L4_EXC_WIDTH = 24
L4_EXC_R = 5
L4_EXC_THRESHOLD = 0.0
L4_EXC_TAU = 1.0
L4_EXC_V_RESET = -75.0
L4_EXC_V_REST = -65.0
L4_EXC_TRACE_TAU = 1.0

# Convolution and STDP parameters
CONV_MODE = "random"
CONV_WEIGHT_SHAPE = (1, 1, 5, 5)
CONV_COEF = 1e-108
CONV_A_PLUS = 1e-107
CONV_A_MINUS = 2e-108

NUM_EPOCHS = 100
######################################################################
# Data Loader
######################################################################
class CustomImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

    def __len__(self):
        return len(self.image_files)
    
    def apply_filter(self, image, filter):
        # convert image to torch tensor
        image_tensor = torch.tensor(image, dtype=torch.float32)
        
        # apply the filter using convolution
        filter_size = filter.shape[0]
        padding = filter_size // 2
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  
        filter = filter.unsqueeze(0).unsqueeze(0) 
        filtered_image = torch.nn.functional.conv2d(image_tensor, filter, padding=padding)
        
        return filtered_image.squeeze().numpy()

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        filter = dog_filter
        image = self.apply_filter(image, dog_filter)
        image = cv2.resize(image, (28, 28))
        filtered_img_norm = (image - image.min()) / (image.max() - image.min())
        
        # convert to torch tensors
        filtered_img_tensor = torch.tensor(filtered_img_norm, dtype=torch.float32)
        spikes_image = poisson_encoder(filtered_img_tensor)
       
        return spikes_image
    

dog_filter = DoGFilter(size=15, sigma_1=1.0, sigma_2=5.0, dtype=torch.float32)
poisson_encoder = Poisson(time_window=POISSON_TIME, ratio=POISSON_RATIO)
transformation = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((42, 42)),
        Conv2dFilter(filters=dog_filter),
        DivideSignPolarity(),
        Poisson(time_window=POISSON_TIME, ratio=POISSON_RATIO),
    ]
)

dataset = CustomImageDataset(image_folder=DATA_ROOT)
dl = DataLoader(dataset, batch_size=4, shuffle=True)


# display the spike train of the image
def show_spike_train(tensor_image):
    spikes = tensor_image.permute(1, 2, 3, 0).reshape(-1, tensor_image.shape[0]).numpy()
    plt.figure(figsize=(12, 8))
    for neuron_idx in range(spikes.shape[0]):
        spike_times = np.where(spikes[neuron_idx])[0]
        plt.scatter(spike_times, np.ones_like(spike_times) * neuron_idx, s=3)
    plt.xlabel('Time')
    plt.ylabel('Neuron Index')
    plt.title('Spike Train Raster Plot')
    plt.show()

##################################################
# initializing neocortex
##################################################
net = Neocortex(dt=DT, device=DEVICE, dtype=DTYPE)
##################################################


##################################################
# input layer
##################################################
input_layer = InputLayer(
    net=net,
    
    input_dataloader=dl,
    have_label=False,
    sensory_size=NeuronDimension(
        depth=1, height=SENSORY_SIZE_HEIGHT, width=SENSORY_SIZE_WIDTH
    ),
    sensory_trace=SENSORY_TRACE_TAU_S,
    instance_duration=POISSON_TIME,
    output_ports={"data_out": (None, [("sensory_pop", {})])}
)



##################################################
# feature extraction layer
##################################################
feature_layer = NeuronGroup(
    net=net,
    size=NeuronDimension(depth=FEATURE_MAPS, height=24, width=24),
    behavior={
        2: SimpleDendriteStructure(),
        3: SimpleDendriteComputation(),
        4: LIF(
                R=L4_EXC_R,
                threshold=L4_EXC_THRESHOLD,
                tau=L4_EXC_TAU,
                v_reset=L4_EXC_V_RESET,
                v_rest=L4_EXC_V_REST,
            ),
        5: SpikeTrace(tau_s=L4_EXC_TRACE_TAU),
        6: KWTA(k=1),
        7: ActivityBaseHomeostasis(
                activity_rate=10,     
                window_size=100,       
                updating_rate=0.1,   
                decay_rate=0.99 
            ),
        8: NeuronAxon(),
        9: EventRecorder("spikes", tag="ng2_evrec"),
    },
)


###################################################
# synaptic connections between layers
##################################################
conv_synapses = SynapseGroup(
    net=net,
    src=input_layer.sensory_pop,
    dst=feature_layer,
     behavior={
                    3: SynapseInit(),
                    4: WeightInitializer(mode=CONV_MODE, weight_shape=CONV_WEIGHT_SHAPE),
                    5: MaxPool2D(),
                    5: Conv2dDendriticInput(current_coef=CONV_COEF),    
                    6: Conv2dSTDP(a_plus=CONV_A_PLUS, a_minus=CONV_A_MINUS),    
                    7: Recorder(variables=["weights"], tag="layers weights"),  
    }
)


##################################################
# Training The model
##################################################
net.initialize()

for epoch in range(NUM_EPOCHS):
    net.simulate_iterations(10)
    print(f"Epoch {epoch}")
