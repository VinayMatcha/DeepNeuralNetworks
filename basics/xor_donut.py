import numpy as np
import matplotlib.pyplot as plt

from basics.EcommerceAnn import forward
# from basics.backprop import derivative_w2,derivative_w1, derivative_b2, derivative_b1


def forward(X, W1, b1, W2, b2):
    Z = X.dot(W1) + b1
    Z = Z * (Z > 0)

    activation = Z.dot(W2) + b2
    Y = 1 / (1 + np.exp(-activation))
    return Y, Z

def derivative_w2(Z, T, Y):
    # Z is (N, M)
    return (T - Y).dot(Z)

def derivative_b2(T, Y):
    return (T - Y).sum()


def derivative_w1(X, Z, T, Y, W2):
    # dZ = np.outer(T-Y, W2) * Z * (1 - Z) # this is for sigmoid activation
    # dZ = np.outer(T-Y, W2) * (1 - Z * Z) # this is for tanh activation
    dZ = np.outer(T-Y, W2) * (Z > 0) # this is for relu activation
    return X.T.dot(dZ)


def derivative_b1(Z, T, Y, W2):
    # dZ = np.outer(T-Y, W2) * Z * (1 - Z) # this is for sigmoid activation
    # dZ = np.outer(T-Y, W2) * (1 - Z * Z) # this is for tanh activation
    dZ = np.outer(T-Y, W2) * (Z > 0) # this is for relu activation
    return dZ.sum(axis=0)

def predict(X, W1, b1, W2, b2):
    Y, _ = forward(X, W1, b1, W2, b2)
    return np.round(Y)

def cost(T ,Y):
    return np.sum(T*np.log(Y) + (1-T)*np.log(1-Y))

def text_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([0, 1, 1, 0])
    W1 = np.random.rand(2, 4)
    b1 =  np.random.rand(4)
    W2 = np.random.rand(4)
    b2 = np.random.rand(1)
    LL = []
    lr = 0.001
    regularization = 0.
    error_rate = None
    for i in range(10000):
        pY, Z = forward(X, W1, b1, W2, b2)
        ll = cost(Y, pY)
        prediction = predict(X, W1, b1, W2, b2)
        er = np.abs(prediction - Y).mean()
        LL.append(ll)
        W2 += lr * (derivative_w2(Z, Y, pY) - regularization * W2)
        W1 += lr * (derivative_w1(X, Z, Y, pY, W2) - regularization * W1)
        b2 += lr * (derivative_b2(Y, pY) - regularization * b2)
        b1 += lr * (derivative_b1(Z, Y, pY, W2) - regularization * b1)
        if i%1000 == 0:
            print(ll)
    print("Final classification % is  ", np.abs(prediction == Y).mean())
    plt.plot(LL)
    plt.show()
    plt.savefig("images/Xor_loglikelihood")



def donut_problem():
    N = 1000
    R_inner = 5
    R_outer = 10
    R1 = np.random.randn(N//2) + R_inner
    theta = 2 * np.pi * np.random.random(N//2)
    X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

    R2 = np.random.randn(N//2) + R_outer
    theta = 2 * np.pi * np.random.random(N//2)
    X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

    X = np.concatenate([X_inner, X_outer])
    Y = np.array([0] * (N//2) + [1] * (N//2))
    n_hidden = 8
    W1 = np.random.randn(2, n_hidden)
    b1 = np.random.randn(n_hidden)
    W2 = np.random.randn(n_hidden)
    b2 = np.random.randn(1)
    LL = []
    lr = 0.00005
    learning_rate= lr
    regularization = 0.2
    error_rate = None
    for i in range(10000):
        pY, Z = forward(X, W1, b1, W2, b2)
        ll = cost(Y, pY)
        prediction = predict(X, W1, b1, W2, b2)
        er = np.abs(prediction - Y).mean()
        LL.append(ll)
        W2 += learning_rate * (derivative_w2(Z, Y, pY) - regularization * W2)
        b2 += learning_rate * (derivative_b2(Y, pY) - regularization * b2)
        W1 += learning_rate * (derivative_w1(X, Z, Y, pY, W2) - regularization * W1)
        b1 += learning_rate * (derivative_b1(Z, Y, pY, W2) - regularization * b1)
        if i % 1000 == 0:
            print(ll)
    print("Final classification % is  ", np.abs( prediction == Y).mean())
    plt.plot(LL)
    plt.show()
    plt.savefig("images/donut_loglikelihood")


if __name__ == '__main__':
    text_xor()
    # donut_problem()