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
		self.N = self.parameter("N", 10)
		ng.v = ng.vector(mode=self.u_rest)  # initialize v with u-rest
		ng.spikes = ng.vector(mode=0)       # save spike times
		# firing
		ng.spike = ng.v >= self.threshold
		ng.v[ng.spike] = self.u_reset
		# ng.population_activity = ng.vector(mode=0)
		# ng.population_activity = 0
		ng.num_spikes = 0
		
	def forward(self, ng):
        # dynamic
		leakage = -(ng.v - self.u_rest)
		currents = self.R * ng.I
		# print("leak: ",  ((leakage + currents) / self.tau) * ng.network.dt)
		ng.v += ((leakage + currents) / self.tau) * ng.network.dt
		# print("v: ",ng.v)
		
		# firing
		ng.spike = ng.v >= self.threshold
		# print(ng.spikes)
		# print(ng.spike)
		ng.num_spikes = torch.sum(ng.spike).item()
		# print(ng.num_spikes)
		# ng.population_activity = ng.vector(mode=0)
		# ng.population_activity += ng.num_spikes / self.N
		# print(ng.population_activity)
		#reset
		ng.v[ng.spike] = self.u_reset