from pymonntorch import Behavior


# set network's dt
class TimeResolution(Behavior):
	def initialize(self, network):
		network.dt = self.parameter("dt", 1.0)