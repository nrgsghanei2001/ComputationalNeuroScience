"""
Dendritic behaviors.
"""
from pymonntorch import Behavior
import torch
import torch.nn.functional as F




class BaseDendriticInput(Behavior):
    """
    Base behavior for turning pre-synaptic spikes to post-synaptic current. It checks for excitatory/inhibitory attributes
    of pre-synaptic neurons and sets a coefficient accordingly.

    Note: weights must be initialize by others behaviors.
          Also, Axon paradigm should be added to the neurons.
          Connection type (Proximal, Distal, Apical) should be specified by the tag
          of the synapse. and Dendrite behavior of the neurons group should access the
          `I` of each synapse to apply them.

    Args:
        current_coef (float): Scalar coefficient that multiplies weights.
    """

    def __init__(self, *args, current_coef=1, **kwargs):
        super().__init__(*args, current_coef=current_coef, **kwargs)

    def initialize(self, synapse):
        """
        Sets the current_type to -1 if the pre-synaptic neurons are inhibitory.

        Args:
            current_coef (float): Strength of the synapse.
        """
        synapse.add_tag(self.__class__.__name__)
        self.current_coef = self.parameter("current_coef", 1)

        self.current_type = (
            -1 if ("GABA" in synapse.src.tags) or ("inh" in synapse.src.tags) else 1
        )

        self.def_dtype = synapse.def_dtype
        synapse.I = synapse.dst.vector(0)

    def calculate_input(self, synapse):
        ...

    def forward(self, synapse):
        synapse.I = (
            self.current_coef * self.current_type * self.calculate_input(synapse)
        )

class MaxPool2D(BaseDendriticInput):
    """
    Max Pooling on Source population spikes.

    Note: Axon paradigm should be added to the neurons.
          Connection type (Proximal, Distal, Apical) should be specified by the tag
          of the synapse. and Dendrite behavior of the neurons group should access the
          `I` of each synapse to apply them.

    Args:
        current_coef (float): Scalar coefficient that multiplies weights.
    """

    def initialize(self, synapse):
        super().initialize(synapse)
        self.output_shape = (synapse.dst.height, synapse.dst.width)

        if synapse.src.depth != synapse.dst.depth:
            raise RuntimeError(
                f"For pooling, source({synapse.src.depth}) and destination({synapse.dst.depth}) should have same depth."
            )

    def calculate_input(self, synapse):
        spikes = synapse.src.axon.get_spike(synapse.src, synapse.src_delay)
        spikes = spikes.view(synapse.src_shape).to(self.def_dtype)
        I = F.adaptive_max_pool2d(spikes, self.output_shape)
        return I.view((-1,))