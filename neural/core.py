import graphlib
import math


class Node:
  def __init__(self, data: float | int, pred: set[Node]):
    self.data, self.pred = float.from_number(data), pred

  def __add__(self, other: Node | float | int):
    other = other if isinstance(other, Node) else Value(other)
    return Addition(self, other)

  def __radd__(self, other: float | int):
    return Addition(Value(other), self)

  def __mul__(self, other: Node | float | int):
    other = other if isinstance(other, Node) else Value(other)
    return Multiplication(self, other)

  def __rmul__(self, other: float | int):
    return Multiplication(Value(other), self)

  def __neg__(self):
    return self * -1.0

  def __sub__(self, other: Node | float | int):
    return self + -other

  def __rsub__(self, other: float | int):
    return Value(other) + -self

  def __pow__(self, other: float | int):
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
  def __init__(self, n: Node, e: float | int):
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
