from pymonntorch import Behavior
import torch


# simple LIF model
class LIF(Behavior):
	def initialize(self, ng):
		# set models parameters
		self.tau = self.parameter("tau")
		self.tau_trace = self.parameter("tau_t", 1.5)
		self.u_rest = self.parameter("u_rest")
		self.u_reset = self.parameter("u_reset")
		self.threshold = self.parameter("threshold")
		self.R = self.parameter("R")
		ng.N = self.parameter("N", 10)
		ng.v = ng.vector(mode=self.u_rest)  # initialize v with u-rest
		ng.spikes = ng.vector(mode=0)       # save spike times
		ng.trace = ng.vector(mode=0)      
		# firing
		ng.spike = ng.v >= self.threshold
		ng.v[ng.spike] = self.u_reset
		ng.population_activity = 0
		ng.num_spikes = 0
		
	def forward(self, ng):
		# firing
		ng.spike = ng.v >= self.threshold
		ng.spike[-1] = False
		ng.num_spikes = torch.sum(ng.spike).item()
		ng.population_activity = ng.num_spikes / ng.N
		ng.trace += -ng.trace/self.tau_trace
		#reset
		ng.v[ng.spike] = self.u_reset
		ng.trace[ng.spike] += 1
        # dynamic
		leakage = -(ng.v - self.u_rest)
		currents = self.R * ng.I
		ng.v += ((leakage + currents) / self.tau) * ng.network.dt

class InputPattern(Behavior):
	def initialize(self, ng):
		# set models parameters
		ng.pattern = self.parameter("pattern")
		ng.pattern2 = self.parameter("pattern2", None)
		self.ch_pattern_time = self.parameter("cpt", 50)
		self.sleep = self.parameter("sleep", 10)
		self.sleep_past = 0
		self.tau_trace = self.parameter("tau_t", 1.5)
		ng.iter = 0
		ng.spike = ng.pattern[ng.iter]  == 1    # save spike times
		ng.num_spikes = 0
		ng.trace = ng.vector(mode=0)
		
	def forward(self, ng):
		# firing
		if ng.iter >= self.ch_pattern_time and self.sleep_past < self.sleep:
			self.sleep_past += 1
			ng.pattern = ng.pattern2
		else:
			ng.spike = ng.pattern[ng.iter] == 1
			ng.spike[-1] = False
			print(ng.spike)
			ng.iter += 1
			ng.trace += -ng.trace/self.tau_trace
			ng.trace[ng.spike] += 1
			
		