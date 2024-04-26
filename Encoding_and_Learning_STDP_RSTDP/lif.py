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
		ng.v[0] = -45  # different initial value for 2 output neurons
		ng.spikes = ng.vector(mode=0)       # save spike times
		ng.trace = ng.vector(mode=0)      
		# firing
		ng.spike = ng.v >= self.threshold
		# ng.spike[-1] = False
		ng.v[ng.spike] = self.u_reset

		
	def forward(self, ng):
		# firing
		ng.spike = ng.v >= self.threshold
		# ng.spike[-1] = False
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
		ng.pn = 0  # pattern number which is applying now
		self.ch_pattern_time = self.parameter("cpt", 50)   # the time to change pattern
		self.sleep = self.parameter("sleep", 10)   # duration between changing patterns
		self.sleep_past = 0  # time that has been passed since sleep mode
		self.tau_trace = self.parameter("tau_t", 1.5)
		ng.iter = 0
		ng.spike = ng.pattern[ng.iter] == 1    # save spike times
		ng.trace = ng.vector(mode=0)
		
	def forward(self, ng):
		# firing
		if ng.iter >= self.ch_pattern_time and self.sleep_past < self.sleep:
			ng.iter += 1
			self.sleep_past += 1
			ng.pattern = ng.pattern2
			ng.spike = ng.pattern[ng.iter] == 2
			# ng.spike[-1] = False
			# print(ng.spike)
			ng.trace += -ng.trace/self.tau_trace
			ng.trace[ng.spike] += 1
			ng.pn = 1
		else:
			ng.spike = ng.pattern[ng.iter] == 1
			# ng.spike[-1] = False
			ng.iter += 1
			ng.trace += -ng.trace/self.tau_trace
			ng.trace[ng.spike] += 1
			
		