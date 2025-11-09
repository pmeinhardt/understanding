from neural.core import backprop, tanh
from neural.nets import MLP

nn = MLP(3, [4, 4, 1], tanh) # 3 inputs, 2 hidden layers of 4 neurons, 1 output

params = nn.params() # gather parameters

xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [1.0, 1.0, -1.0], [0.5, 1.0, 1.0]] # inputs
ys = [1.0, -1.0, 1.0, -1.0] # expected outputs

for k in range(100):
  # forward pass
  pred = [nn(x)[0] for x in xs] # type: ignore[arg-type]

  # compute loss
  loss = sum((target - actual)**2 for (target, actual) in zip(ys, pred))

  # compute gradients
  grads = backprop(loss)

  # perform gradient descent
  for p in params:
    p.data -= 0.1 * grads[p]

  # print loss and predictions
  print(f"{k:-2}: {loss.data}, pred: {[p.data for p in pred]}")

# final predictions
print([p.data for p in pred])
