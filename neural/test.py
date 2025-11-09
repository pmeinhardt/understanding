from unittest import main, TestCase

from .core import backprop, tanh, Value


class TestPropagation(TestCase):
  def test(self):
    x1 = Value(2.0)
    x2 = Value(0.0)

    w1 = Value(-3.0)
    w2 = Value(1.0)

    b = Value(6.8813735870195432)

    n = (x1 * w1) + (x2 * w2) + b
    o = tanh(n)

    self.assertEqual(o.data, 0.7071067811865476)


class TestBackward(TestCase):
  def test(self):
    x1 = Value(2.0)
    x2 = Value(0.0)

    w1 = Value(-3.0)
    w2 = Value(1.0)

    b = Value(6.8813735870195432)

    n = (x1 * w1) + (x2 * w2) + b
    o = tanh(n)

    grads = backprop(o)

    self.assertAlmostEqual(grads[n], 0.5)
    self.assertAlmostEqual(grads[b], 0.5)

    self.assertAlmostEqual(grads[w1], 1.0)
    self.assertAlmostEqual(grads[w2], 0.0)

    self.assertAlmostEqual(grads[x1], -1.5)
    self.assertAlmostEqual(grads[x2], 0.5)


if __name__ == '__main__':
  main()
