from pymonntorch import Behavior
import numpy as np
import random


# set constant current for models
class ConstantCurrent(Behavior):
    def initialize(self, ng):
        self.value = self.parameter("value", None)
        ng.I = ng.vector(self.value)   # initialize I vector with given value
        ng.inputI = ng.vector(self.value)   # initialize I vector with given value

    def forward(self, ng):
        ng.I = ng.vector(self.value)    # at each iteration the current is the constant value
        ng.inputI = ng.vector(self.value)    # at each iteration the current is the constant value


# set step current. Start with zero and from t0 to the end I is constant value
class StepCurrent(Behavior):
	def initialize(self, ng):
		self.value = self.parameter("value")
		self.t0 = self.parameter("t0")
		ng.I = ng.vector(mode=0)

	def forward(self, ng):
		if ng.network.iteration * ng.network.dt >= self.t0:         # after t0 iterations set the current
			ng.I = ng.vector(mode=self.value) * ng.network.dt
			

# set current as staircase. In each time interval I is constant
class StaircaseCurrent(Behavior):
	def initialize(self, ng):
		self.value = self.parameter("value")
		self.t = self.parameter("t")
		ng.I = ng.vector(mode=0)

	def forward(self, ng):
		if ng.network.iteration * ng.network.dt % self.t == 0:      # if networks iterations reached the end 
			ng.I += ng.vector(mode=self.value) * ng.network.dt      # of time interval, change the constant value
			


# set a sinusoidal I
class SinCurrent(Behavior):
	def initialize(self, ng):
		self.amplitude = self.parameter("amplitude")
		self.frequency = self.parameter("frequency")
		ng.I = ng.vector(mode=0)

	def forward(self, ng):
		current_value = self.amplitude * np.sin(2 * np.pi * self.frequency * ng.network.iteration * ng.network.dt)  # find sin value
		ng.I += current_value 
		


# set a sinusoidal current with adding random noise to it
class NoisyCurrent(Behavior):
	def initialize(self, ng):
		self.amplitude = self.parameter("amplitude")
		self.frequency = self.parameter("frequency")
		ng.I = ng.vector(mode=0)

	def forward(self, ng):
		noise = random.random()   # add 2 random noise to current value
		noise2 = random.random()
		current_value = self.amplitude * np.sin(2 * np.pi * self.frequency * ng.network.iteration * ng.network.dt * noise)
		ng.I += current_value * noise2


# set a constant current with adding random noise to it
class NoisyConstantCurrent(Behavior):
	def initialize(self, ng):
		self.value = self.parameter("value")
		ng.I = ng.vector(mode=self.value)

	def forward(self, ng):
		ng.I = ng.vector(self.value)
		noise = random.randint(-10, 10)   # add random noise to current value
		ng.I += noise


# set a staircase current with adding random noise to it
class NoisyStairCurrent(Behavior):
	def initialize(self, ng):
		self.value0 = self.parameter("value0")   # first value of I
		self.value1 = self.parameter("value1")   # second value of I
		self.currentval = self.value0
		self.t0 = self.parameter("t0")           
		self.t = self.parameter("t")
		ng.I = ng.vector(mode=self.value0)

	def forward(self, ng):
		
		if (ng.network.iteration * ng.network.dt) >= self.t0:
			if (ng.network.iteration * ng.network.dt) % self.t == 0:         # after t iterations reset the current
				ng.I = ng.vector(mode=self.currentval)
				ng.I += ng.vector(mode=self.value1) * ng.network.dt
				self.currentval += self.value1
				noise = random.randint(-10, 10)   # add random noise to current value
				ng.I += noise
			else:
				ng.I = ng.vector(mode=self.currentval)
		else:
			ng.I = ng.vector(mode=self.value0)

		
		