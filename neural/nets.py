from functools import reduce
from itertools import pairwise
from typing import Callable

import random

from .core import Node, Value, backprop, tanh


class Neuron:
  def __init__(self, nin: int, phi: Callable[[Node], Node]):
    self.w = [Value(random.uniform(-1.0, 1.0)) for _ in range(nin)] # weights
    self.b = Value(random.uniform(-1.0, 1.0)) # bias
    self.f = phi # activation function

  def __call__(self, x: list[Node | float]):
    act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
    return self.f(act)

  def params(self):
    return self.w + [self.b]


class Layer:
  def __init__(self, nin: int, nout: int, phi: Callable[[Node], Node]):
    self.neurons = [Neuron(nin, phi) for _ in range(nout)]

  def __call__(self, x: list[Node | float]):
    return [n(x) for n in self.neurons]

  def __iter__(self):
    return iter(self.neurons)

  def __len__(self):
    return len(self.neurons)

  def __getitem__(self, index: int):
    return self.neurons[index]

  def params(self):
    return [p for n in self.neurons for p in n.params()]


class MLP:
  def __init__(self, nin: int, nouts: list[int], phi: Callable[[Node], Node]):
    self.layers = [Layer(nin, nout, phi) for nin, nout in pairwise([nin] + nouts)]

  def __call__(self, x: list[Node | float]):
    x = [xi if isinstance(xi, Node) else Value(xi) for xi in x]
    return reduce(lambda y, layer: layer(y), self.layers, x)

  def __iter__(self):
    return iter(self.layers)

  def __len__(self):
    return len(self.layers)

  def __getitem__(self, index: int):
    return self.layers[index]

  def params(self):
    return [p for l in self.layers for p in l.params()]
