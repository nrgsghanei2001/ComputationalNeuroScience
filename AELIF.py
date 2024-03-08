from pymonntorch import *

# Adaptive Exponential LIF
class AELIF(Behavior):
    def initialize(self, ng):
        self.tau_m = self.parameter("tau_m")
        self.tau_w = self.parameter("tau_w")
        self.u_rest = self.parameter("u_rest")
        self.u_reset = self.parameter("u_reset")
        self.threshold = self.parameter("threshold")
        self.delta_t = self.parameter("delta_t")
        self.R = self.parameter("R")
        self.theta_rh = self.parameter("theta_rh")
        self.alpha = self.parameter("alpha")
        self.beta = self.parameter("beta")

        ng.v = ng.vector(mode=self.u_rest)
        ng.spikes = ng.vector(mode=0)
        ng.w = ng.vector(mode=0)
        ng.num_spike = ng.vector(mode=0)


    def forward(self, ng):
        # firing
        if ng.v > self.threshold:
            ng.num_spike += 1

        ng.spike = ng.v >= self.threshold

        # reset
        ng.v[ng.spike] = self.u_reset

        # dynamic
        linear = -(ng.v - self.u_rest)
        ri = self.R * ng.I
        v1 = (ng.v - self.theta_rh) / self.delta_t
        F = self.delta_t * np.exp(v1)
        u = linear + F + ri
        dvdt = u - self.R * ng.w
        ng.v += dvdt / self.tau_m
        
        # adaptation
        adaptation = self.beta * self.tau_w * ng.num_spike 
        sub_adaptation = self.alpha * (ng.v - self.u_rest)
        dAdt = sub_adaptation + adaptation - ng.w
        ng.w = dAdt / self.tau_w
