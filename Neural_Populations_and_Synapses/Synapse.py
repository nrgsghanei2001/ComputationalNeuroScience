from pymonntorch import Behavior
import torch


class SynFun(Behavior):
	def initialize(self, sg):
		sg.W = sg.matrix(mode="normal(0.5, 0.3)")
		sg.I = sg.dst.vector()

	def forward(self, sg):
		sg.I = torch.sum(sg.W[sg.src.spike], axis=0)

class InpSyn(Behavior):
	def forward(self, ng):
		for syn in ng.afferent_synapses["All"]:
			ng.I += syn.I 
