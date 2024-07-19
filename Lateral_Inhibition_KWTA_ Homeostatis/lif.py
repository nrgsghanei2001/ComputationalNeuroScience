from pymonntorch import Behavior
import torch


# simple LIF model
class LIF(Behavior):
	def initialize(self, ng):
		# set models parameters
		self.tau = self.parameter("tau")
		self.tau_trace = self.parameter("tau_t", 1.5)
		self.u_rest = self.parameter("u_rest")
		self.v_reset = self.parameter("v_reset")
		ng.v_reset = self.v_reset
		self.threshold = self.parameter("threshold")
		ng.threshold =  ng.vector(mode=self.threshold)
		self.R = self.parameter("R")
		ng.N = self.parameter("N", 10)
		ng.v = ng.vector(mode=self.u_rest)  # initialize v with u-rest
		ng.v[0] = -60  # different initial value for 2 output neurons
		ng.spikes = ng.vector(mode=0)       # save spike times
		ng.spike = ng.vector(mode=0)       # save spike times
		ng.trace = ng.vector(mode=0)      
		# firing
		ng.spike = ng.v >= self.threshold
		ng.spikes = ng.v >= self.threshold
		# ng.spike[-1] = False
		ng.v[ng.spike] = ng.v_reset
		ng.iteration = 0

		
	def forward(self, ng):
		# firing
		ng.spike = ng.v >= ng.threshold
		ng.spikes = ng.v >= ng.threshold
		# ng.spike = ng.v >= self.threshold
		# ng.spike[-1] = False
		ng.trace += -ng.trace/self.tau_trace
		# for x in range(ng.trace.shape[0]):
		# 	for y in range(ng.trace.shape[1]):
		# 		if ng.trace[x][y] < 1e-3:
		# 			ng.trace[x][y] = 0 
		#reset
		ng.v[ng.spike] = self.v_reset
		ng.trace[ng.spike] += 1
        # dynamic
		leakage = -(ng.v - self.u_rest)
		currents = self.R * ng.I
		ng.v += ((leakage + currents) / self.tau) * ng.network.dt
		ng.iteration += 1



class InputPattern(Behavior):
	def initialize(self, ng):
		# set models parameters
		ng.pattern = self.parameter("pattern")
		ng.pattern2 = self.parameter("pattern2", None)
		ng.pattern3 = self.parameter("pattern3", None)
		ng.pattern4 = self.parameter("pattern4", None)
		ng.pattern5 = self.parameter("pattern5", None)
		ng.num_rep = self.parameter("nume_rep", 100)
		ng.pn = 0  # pattern number which is applying now
		self.ch_pattern_time = self.parameter("cpt", 50)   # the time to change pattern
		self.sleep = self.parameter("sleep", 10)   # duration between changing patterns
		self.sleep_past = 0  # time that has been passed since sleep mode
		self.tau_trace = self.parameter("tau_t", 1.5)
		ng.iter = 0
		ng.spike = ng.pattern[ng.iter] == 1    # save spike times
		ng.trace = ng.vector(mode=0)
		ng.v_reset = -70
		ng.v = ng.vector(mode=-65)
		ng.threshold = -55
		self.chpn = [40, 100, 160, 220]
		self.patterns = [ng.pattern, ng.pattern2, ng.pattern3, ng.pattern4, ng.pattern5]
		ng.spikes = ng.vector(mode=0)

	def forward(self, ng):
		# firing
		# if ng.iter == ng.num_rep:
		# 	ng.iter = 0
		# 	self.sleep_past = 0
		# 	ng.pn = 0

		# if ng.iter in self.ch_pattern_time and self.sleep_past < self.sleep:
		# 	ng.iter += 1
		# 	self.sleep_past += 1
		# 	ng.pattern = ng.pattern2
		# 	ng.spike = ng.pattern[ng.iter] == 1
		# 	ng.v[ng.spike] = -55
		# 	# ng.spike[-1] = False
		# 	# print(ng.spike)
		# 	ng.trace += -ng.trace/self.tau_trace
		# 	ng.trace[ng.spike] += 1
		# 	ng.pn = 1
		if ng.iter in self.chpn:
			ng.iter += 1
			self.sleep_past += 1
			ng.pattern = self.patterns[ng.pn]
			ng.spike = ng.pattern[ng.iter] == 1
			ng.spikes = ng.pattern[ng.iter] == 1
			ng.v[ng.spike] = -55
			# ng.spike[-1] = False
			# print(ng.spike)
			ng.trace += -ng.trace/self.tau_trace
			ng.trace[ng.spike] += 1
			ng.pn += 1
		elif self.sleep_past < self.sleep and self.sleep_past != 0:
			ng.iter += 1
			self.sleep_past += 1
			ng.spike = ng.pattern[ng.iter] == 1
			ng.spikes = ng.pattern[ng.iter] == 1
			ng.v[ng.spike] = -55
			# ng.spike[-1] = False
			# print(ng.spike)
			ng.trace += -ng.trace/self.tau_trace
			ng.trace[ng.spike] += 1

		else:
			self.sleep_past = 0
			ng.spike = ng.pattern[ng.iter] == 1
			ng.spikes = ng.pattern[ng.iter] == 1
			ng.v[ng.spike] = -55
			# ng.spike[-1] = False
			ng.iter += 1
			ng.trace += -ng.trace/self.tau_trace
			ng.trace[ng.spike] += 1
			
		