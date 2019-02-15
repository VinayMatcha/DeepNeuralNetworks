import numpy as np
import matplotlib.pyplot as plt
np.seterr(divide='ignore', invalid='ignore')
# a = np.random.randn(100, 5)
# soft = np.exp(a)
# soft = soft/soft.sum(axis=1, keepdims=True)
# print(soft.sum())

def sigmoid(a):
    return 1 / (1+np.exp(-a))

def forward(X, W1, b1, W2, b2):
    Z1 = sigmoid(X.dot(W1) + b1)
    expa = np.exp(Z1.dot(W2) + b2)
    expa = expa/expa.sum(axis=1, keepdims= True)
    return expa, Z1

def classification(T, Y):
    count= 0
    for i in range(len(Y)):
        if Y[i] == T[i]:
            count+=1
    return count/len(Y)

def cost(T, Y):
    tot =  T * np.log(Y)
    return tot.sum()

def derivative_w2(Z, T, Y):
    val = Z.T.dot(T-Y)
    return val

def derivative_b2(T, Y):
    return (T-Y).sum(axis=0)

def derivative_w1(X, Z, T, Y, W2):
    dz = (T-Y).dot(W2.T)*Z*(1-Z)
    return X.T.dot(dz)

def derivative_b1(T, Y, W2, Z):
    return (T-Y.dot(W2.T)*Z*(1-Z)).sum(axis=0)





def main():
    N = 500
    D = 2
    M = 3
    K = 3
    X1 = np.random.randn(N, 2) + np.array([0, -2])
    X2 = np.random.randn(N, 2) + np.array([2, 2])
    X3 = np.random.randn(N, 2) + np.array([-2, 2])
    X = np.vstack((X1, X2, X3))
    X = X
    print(X.shape)
    Y = np.array([0] * N + [1] * N + [2] * N)
    N = len(Y)
    T = np.zeros((N, K))
    for i in range(N):
        T[i, Y[i]] = 1
    plt.scatter(X[:,0], X[:,1], c = Y, s=100, alpha = 0.5)
    plt.show()
    plt.savefig("images/3clusterdata")
    lr = 1e-3
    costs = []
    W1 = np.random.randn(D, M)
    b1 = np.random.randn(M)
    W2 = np.random.randn(M, K)
    b2 = np.random.randn(K)
    for epoch in range(1000):
        output, hidden = forward(X, W1, b1, W2, b2)
        if epoch%100 == 0:
            c = cost(T, output)
            p = np.argmax(output, axis=1)
            costs.append(c)
            print("costs is ", c, " eff id ", classification(Y, p))

        W2 += lr * derivative_w2(hidden, T, output)
        b2 += lr * derivative_b2(T, output)
        W1 += lr * derivative_w1(X, hidden, T, output, W2)
        b2 += lr * derivative_b1(T, output, W2, hidden)
    plt.plot(costs)
    plt.show()
    plt.ylabel("negative likelihood Cost value")
    plt.xlabel("number of iterations")
    plt.savefig("images/backpropCost")


if __name__ == '__main__':
    main()