from functools import reduce
from itertools import pairwise

import graphlib
import math
import random


class Node:
  def __init__(self, data: float, pred: Set[Node]):
    self.data, self.pred = data, pred

  def __add__(self, other: Node | float):
    other = other if isinstance(other, Node) else Value(other)
    return Addition(self, other)

  def __mul__(self, other: Node | float):
    other = other if isinstance(other, Node) else Value(other)
    return Multiplication(self, other)

  def __rmul__(self, other: float):
    return Multiplication(Value(other), self)

  def __neg__(self):
    return self * -1.0

  def __sub__(self, other: Node | float):
    return self + -other

  def __pow__(self, other: float):
    return Power(self, other)

  def backprop(self, grad: float):
    raise NotImplementedError("subclasses should implement this method")


class Value(Node):
  def __init__(self, data: float):
    super().__init__(data, set())

  def __repr__(self):
    return f"{self.data}"

  def backprop(self, grad: float):
    return {}


class Addition(Node):
  def __init__(self, l: Node, r: Node):
    super().__init__(l.data + r.data, {l, r})
    self.l, self.r = l, r

  def __repr__(self):
    return f"({self.l})+({self.r})"

  def backprop(self, grad: float):
    return {self.l: grad, self.r: grad}


class Multiplication(Node):
  def __init__(self, l: Node, r: Node):
    super().__init__(l.data * r.data, {l, r})
    self.l, self.r = l, r

  def __repr__(self):
    return f"({self.l})*({self.r})"

  def backprop(self, grad: float):
    return {self.l: self.r.data * grad, self.r: self.l.data * grad}


class Power(Node):
  def __init__(self, n: Node, e: int | float):
    super().__init__(n.data**e, {n})
    self.n, self.e = n, e

  def __repr__(self):
    return f"({self.b}**{self.e})"

  def backprop(self, grad: float):
    return {self.n: self.e * self.n.data**(self.e - 1) * grad}


class Tanh(Node):
  def __init__(self, n: Node):
    super().__init__(math.tanh(n.data), {n})
    self.n = n

  def __repr__(self):
    return f"tanh({self.n})"

  def backprop(self, grad: float):
    return {self.n: (1 - self.data**2) * grad}


def tanh(node: Node):
  return Tanh(node)


def topo(start: Node):
  stack = [start]
  graph = {}

  while len(stack) > 0:
    node = stack.pop()
    graph[node] = node.pred
    stack += list(node.pred)

  sorter = graphlib.TopologicalSorter(graph)
  return sorter.static_order()


def backprop(start: Node):
  grads = {start: 1.0}

  ordered = reversed(list(topo(start)))

  for node in ordered:
    for child, grad in node.backprop(grads[node]).items():
      if child in grads:
        grads[child] += grad
      else:
        grads[child] = grad

  return grads


class Neuron:
  def __init__(self, nin: int):
    self.w = [Value(random.uniform(-1.0, 1.0)) for _ in range(nin)] # weights
    self.b = Value(random.uniform(-1.0, 1.0)) # bias

  def __call__(self, x: [Node]):
    act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
    return tanh(act)

  def params(self):
    return self.w + [self.b]


class Layer:
  def __init__(self, nin: int, nout: int):
    self.neurons = [Neuron(nin) for _ in range(nout)]

  def __call__(self, x: [Node]):
    return [n(x) for n in self.neurons]

  def __iter__(self):
    return iter(self.neurons)

  def __len__(self):
    return len(self.neurons)

  def __get__(self, index: int):
    return self.neurons[index]

  def params(self):
    return [p for n in self.neurons for p in n.params()]


class MLP:
  def __init__(self, nin: int, nouts: [int]):
    self.layers = [Layer(nin, nout) for nin, nout in pairwise([nin] + nouts)]

  def __call__(self, x: [Node | float]):
    x = [xi if isinstance(xi, Node) else Value(xi) for xi in x]
    return reduce(lambda y, layer: layer(y), self.layers, x)

  def __iter__(self):
    return iter(self.layers)

  def __len__(self):
    return len(self.layers)

  def __get__(self, index: int):
    return self.layers[index]

  def params(self):
    return [p for l in self.layers for p in l.params()]


if __name__ == '__main__':
  nn = MLP(3, [4, 4, 1]) # 3 inputs, 2 hidden layers of 4 neurons, 1 output

  params = nn.params() # gather parameters

  xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [1.0, 1.0, -1.0], [0.5, 1.0, 1.0]] # inputs
  ys = [1.0, -1.0, 1.0, -1.0] # expected outputs

  for k in range(100):
    # forward pass
    pred = [nn(x)[0] for x in xs]

    # compute loss
    loss = sum(((Value(target) - actual)**2 for (target, actual) in zip(ys, pred)), Value(0.0))

    # compute gradients
    grads = backprop(loss)

    # perform gradient descent
    for p in params:
      p.data -= 0.1 * grads[p]

    # print loss and predictions
    print(f"{k:-2}: {loss.data}, pred: {[p.data for p in pred]}")

  # final predictions
  print([p.data for p in pred])
