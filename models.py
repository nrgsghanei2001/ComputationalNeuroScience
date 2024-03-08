from pymonntorch import *




# simple LIF model
class LIF(Behavior):
	def initialize(self, ng):
		# set models parameters
		self.tau = self.parameter("tau")
		self.u_rest = self.parameter("u_rest")
		self.u_reset = self.parameter("u_reset")
		self.threshold = self.parameter("threshold")
		self.R = self.parameter("R")
		
		ng.v = ng.vector(mode=self.u_rest)  # initialize v with u-rest
		ng.spikes = ng.vector(mode=0)       # save spike times
		

	def forward(self, ng):
        # dynamic
		leakage = -(ng.v - self.u_rest)
		currents = self.R * ng.I
		ng.v += ((leakage + currents) / self.tau) * ng.network.dt
		
		# firing
		ng.spike = ng.v >= self.threshold

		#reset
		ng.v[ng.spike] = self.u_reset
				



# Exponential LIF model(ELIF)
class ELIF(Behavior):
    def initialize(self, ng):
        # initializing model's parameters
        self.tau = self.parameter("tau")
        self.u_rest = self.parameter("u_rest")
        self.u_reset = self.parameter("u_reset")
        self.theta_rh = self.parameter("theta_rh")
        self.threshold = self.parameter("threshold")
        self.delta_t = self.parameter("delta_t")
        self.R = self.parameter("R")

        ng.v = ng.vector(mode=self.u_rest)	
        ng.spikes = ng.vector(mode=0)
        

    def forward(self, ng):
        # dynamic
        linear = -(ng.v - self.u_rest)
        v1 = (ng.v - self.theta_rh) / self.delta_t
        F = self.delta_t * np.exp(v1)
        ri = self.R * ng.I
        membrane_potential_change = linear + F + ri
        ng.v += (membrane_potential_change / self.tau) * ng.network.dt

        # firing
        ng.spike = ng.v >= self.threshold

        # reset
        ng.v[ng.spike] = self.u_reset