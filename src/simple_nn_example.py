from nn import MLP

if __name__ == '__main__':
    nn = MLP(3, [4, 4, 1])

    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ]

    ys = [1.0, -1.0, -1.0, 1.0]
    ypred = [nn(x) for x in xs]
    loss = sum((yout - ygt) ** 2 for yout, ygt in zip(ypred, ys))
    print(loss)
    print(f"Loss before SDG = {loss.data}")
    for i in range(100):
        for p in nn.parameters():
            p.grad = 0
        loss.backward()
        for p in nn.parameters():
            p.data -= 0.01 * p.grad
        ypred = [nn(x) for x in xs]
        loss = sum((yout - ygt) ** 2 for yout, ygt in zip(ypred, ys))
        print(f"Loss after {i} iteration(s) = {loss.data}")
    print(f"Expected:          {ys}")
    print(f"Network predicted: {[y.data for y in ypred]}")