"""Equivalent batched execution of the existing HOI-TG graph convolutions.

No parameter names, graph edges, architecture, precision or batch sizes change.
The sparse human graph acts independently on the batch/channel columns.
"""
import types
import torch
from gcanet.reconstruction.models.gcc import GraphConvolution as HumanGraphConvolution
from gcanet.reconstruction.models.ogcc import GraphConvolution as ObjectGraphConvolution


class StaticGraphMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, adjacency, transpose, dense):
        ctx.save_for_backward(transpose)
        return torch.sparse.mm(adjacency, dense)

    @staticmethod
    def backward(ctx, gradient):
        transpose, = ctx.saved_tensors
        return None, None, torch.sparse.mm(transpose, gradient)


def human_graph_forward(self, x):
    if x.ndim == 2:
        out = torch.matmul(self.adjmat, torch.matmul(x, self.weight))
    else:
        support = torch.matmul(x, self.weight)
        batch, nodes, channels = support.shape
        columns = support.permute(1, 0, 2).reshape(nodes, batch*channels)
        out = StaticGraphMM.apply(self.adjmat, self._transpose_adjacency, columns)
        out = out.reshape(nodes, batch, channels).permute(1, 0, 2)
    return out+self.bias if self.bias is not None else out


def object_graph_forward(self, x, graph):
    out = torch.bmm(graph, torch.matmul(x, self.weight))
    return out+self.bias if self.bias is not None else out


def apply_throughput_fixes(model):
    count = dict(human=0, object=0)
    for module in model.modules():
        if isinstance(module, HumanGraphConvolution):
            module.adjmat = module.adjmat.coalesce()
            module._transpose_adjacency = module.adjmat.t().coalesce()
            module.forward = types.MethodType(human_graph_forward, module)
            count['human'] += 1
        elif isinstance(module, ObjectGraphConvolution):
            module.forward = types.MethodType(object_graph_forward, module)
            count['object'] += 1
    model.throughput_fixes = dict(version='batched_graph_coalesced_v1', modules=count,
                                  precision='unchanged FP32; frozen DINO FP16')
    return model.throughput_fixes
