# Computational Neuroscience Project

## Overview

This project is a comprehensive implementation of various computational neuroscience models and techniques using the `Pymonntorch` and `CoNex` libraries.
The goal is to simulate and analyze neural behaviors and mechanisms, providing a toolkit for exploring advanced concepts in computational neuroscience.

## Features

- **LIF (Leaky Integrate-and-Fire) Neurons**: Simulate basic neuron dynamics using the LIF model.
- **ELIF (Exponential Leaky Integrate-and-Fire) Neurons**: Enhanced neuron model with exponential terms to capture more complex neuronal behavior.
- **AELIF (Adaptive Exponential Leaky Integrate-and-Fire) Neurons**: Adaptive neuron model that incorporates adaptation mechanisms.
- **Decision Making**: Model decision-making processes within neural populations.
- **Neural Populations**: Simulate groups of neurons to study population dynamics.
- **Synapse Groups**: Implement and manage connections between neurons, with different types of synapses.
- **Time to First Spike Encoding**: Encode information based on the timing of the first spike in a neural response.
- **Poisson Encoding**: Encode numerical values into spike trains using a Poisson distribution.
- **Numerical Values Encoding**: Directly encode numerical values into neural activity.
- **STDP (Spike-Timing-Dependent Plasticity)**: Implement learning rules based on the timing of pre and post-synaptic spikes.
- **RSTDP (Reward-Modulated Spike-Timing-Dependent Plasticity)**: Extend STDP with reward modulation to simulate learning mechanisms.
- **Lateral Inhibition**: Model inhibitory interactions between neurons to enhance contrast in neural responses.
- **K-Winners-Take-All (kWTA)**: Implement a competitive mechanism where only a subset of neurons is allowed to fire.
- **Homeostasis**: Ensure stable neural activity over time by adjusting neural parameters.
- **DoG (Difference of Gaussians) Filter**: Implement DoG filters for image processing and edge detection.
- **Gabor Filters**: Use Gabor filters for feature extraction and texture analysis in images.
- **Image Feature Extraction**: Implement convolutional layers and max pooling for feature extraction from images.

## Installation

To use this project, install the necessary dependencies.

```bash
pip install cnrl-conex
```

```bash
pip install pymonntorch
```

