from pymonntorch import Behavior
import torch


# simple LIF model
class LIF(Behavior):
	def initialize(self, ng):
		# set models parameters
		self.tau = self.parameter("tau")
		self.u_rest = self.parameter("u_rest")
		self.u_reset = self.parameter("u_reset")
		self.threshold = self.parameter("threshold")
		self.R = self.parameter("R")
		k = self.parameter("v_init", default="normal(0.3, 0.05)")
		ng.v = ng.vector(mode=k)  # initialize v with u-rest
		ng.spikes = ng.vector(mode=0)       # save spike times
		# firing
		ng.spike = ng.v >= self.threshold
		ng.v[ng.spike] = self.u_reset
		
	def forward(self, ng):
        # dynamic
		leakage = -(ng.v - self.u_rest)
		currents = self.R * ng.I
		ng.v += ((leakage + currents) / self.tau) * ng.network.dt
		
		# firing
		ng.spike = ng.v >= self.threshold
		#reset
		ng.v[ng.spike] = self.u_reset