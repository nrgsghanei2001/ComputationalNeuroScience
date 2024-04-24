from pymonntorch import Behavior
import torch


class STDP(Behavior):
    def initialize(self, sg):
        self.lr = self.parameter("lr", [5, 5])
        self.weight_decay = self.parameter("wd", 2)


    def forward(self, sg):

        mask = torch.ones(*sg.W.size())
        src_s = self.mask_spike_trace(sg.src.spike, mask)
        dst_s = sg.dst.spike * mask
        src_t = self.mask_spike_trace(sg.src.trace, mask)
        dst_t = sg.dst.trace * mask
        
        A1 = self.lr[0] * src_s * dst_t
        A2 = self.lr[1] * dst_s * src_t
        sg.W += -A1 + A2
        print(sg.W)

    def mask_spike_trace(self, inp, mask):
        if len(inp.size()) > 1:
            inp = torch.t(inp)
        else:
            inp = inp.view(inp.size(0), -1)

        return inp * mask




# class STDP2(LearningRule):
#     """
#     Spike-Time Dependent Plasticity learning rule.
#     Implement the dynamics of STDP learning rule.You might need to implement\
#     different update rules based on type of connection.
#     """

#     def __init__(
#         self,
#         connection: AbstractConnection,
#         lr: Optional[Union[float, Sequence[float]]] = None,
#         weight_decay: float = 0.,
#         **kwargs
#     ) -> None:
#         super().__init__(
#             connection=connection,
#             lr=lr,
#             weight_decay=weight_decay,
#             **kwargs
#         )
#         """
#         TODO.
#         Consider the additional required parameters and fill the body\
#         accordingly.
#         """
#         self.connection = connection
#         self.lr = lr
#         self.weight_decay
        
#         self.dt = kwargs.get("dt", 1.)

#     def update(self, **kwargs) -> None:
#         """
#         TODO.
#         Implement the dynamics and updating rule. You might need to call the\
#         parent method.
#         """
#         mask = torch.ones(*self.connection.w.size())
        
#         pre_s = self.connection.pre.s

#         if len(pre_s.size()) > 1:
#             pre_s = torch.t(pre_s)
#         else:
#             pre_s = pre_s.view(pre_s.size(0), -1)
#         pre_s = pre_s * mask
        
#         post_traces = self.connection.post.traces * mask
#         # print(post_traces)
#         p1 = self.lr[0] * pre_s * post_traces
        
        
#         post_s = self.connection.post.s * mask
        
#         pre_traces = self.connection.pre.traces
#         if len(pre_traces.size()) > 1:
#             pre_traces = torch.t(pre_traces)
#         else:
#             pre_traces = pre_traces.view(pre_traces.size(0), -1)
#         pre_traces = pre_traces * mask
#         # print(pre_traces)
#         p2 = self.lr[1] * post_s * pre_traces
#         # print(self.lr[1])
#         self.connection.w += ((-p1) + p2) * self.dt
#         self.connection.w = torch.clamp(self.connection.w, self.connection.wmin, self.connection.wmax)

#         super().update()
