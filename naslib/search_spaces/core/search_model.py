import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from naslib.search_spaces.nasbench1shot1.utils import PRIMITIVES
from naslib.search_spaces.core.primitives import ConvBnRelu, ReLUConvBN
from naslib.search_spaces.core.operations import MixedOp
from metaclasses import MetaCell, MetaModel


class Cell(MetaCell):

  def __init__(self, graph, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    self.reduction = reduction
    self.graph = graph

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self.graph.num_intermediate_nodes):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride)
        self._ops.append(op)

  def forward(self, s0, s1, alphas):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    if cell.reduction:
      weights = self.graph._nonlinearity(alpha[1], dim=-1)
    else:
      weights = self.graph._nonlinearity(alphas[0], dim=-1)

    states = [s0, s1]
    offset = 0
    for i in range(self.graph.num_intermediate_nodes):
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)


class Network(MetaModel):
  #TODO: change this later with a classmethod that reads the config_file
  def __init__(self, graph, config_file):
    super(Network, self).__init__()
    self.graph = graph
    self.config = load_config(config_file)

    self._C = self.config['init_channels']
    self._num_classes = self.config['num_classes']
    self._layers = self.config['layers']
    self._criterion = eval('nn.'+self.config['criterion'])()
    self._multiplier = self.config['multiplier']
    self.graph._nonlinearity = F.softmax

    C_curr = self._multiplier*self._C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )

    C_prev_prev, C_prev, C_curr = C_curr, C_curr, self._C
    self.cells = nn.ModuleList()

    reduction_prev = False
    for i in range(self._layers):
      if i in [self._layers//3, 2*self._layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False

      cell = Cell(self.graph, self._multiplier, C_prev_prev, C_prev, C_curr,
                  reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, self._multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, self._num_classes)

    self._initialize_alphas()

  def forward(self, input):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self._layers):
      s0, s1 = s1, cell(s0, s1, self._arch_parameters)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def new(self):
    model_new = Network(self.graph, self.config).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target)

  def _initialize_alphas(self):
    k = sum(1 for i in range(self.graph.num_intermediate_nodes) for n in
            range(2+i))
    num_ops = len(self.graph._primitives)

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  #TODO change this in order to be consistent with the networkx graph representation
  def genotype(self):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self.graph.num_intermediate_nodes):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(2+self.graph.num_intermediate_nodes - self._multiplier,
                   self.graph.num_intermediate_nodes+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

