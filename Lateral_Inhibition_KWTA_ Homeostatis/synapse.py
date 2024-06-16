from pymonntorch import Behavior
import torch
import random

class SynFun(Behavior):
	def initialize(self, sg):
		"hello"
		sg.W = sg.matrix(mode=0)
		sg.I = sg.dst.vector()
		self.is_inhibitory = self.parameter("is_inhibitory", False)

		for i in range(sg.W.shape[0]):
			for j in range(sg.W.shape[1]):
				sg.W[i][j] = random.random()
				print(sg.W[i][j])
		print(sg.W)
		print("bye")

	def forward(self, sg):
		sg.I = torch.sum(sg.W[sg.src.spike], axis=0)
		if self.is_inhibitory:
			sg.I *= -1


class InpSyn(Behavior):	
	def forward(self, ng):
		for syn in ng.afferent_synapses["All"]:
			ng.I += (syn.I / ng.N)
