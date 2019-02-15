import numpy as np
import matplotlib.pyplot as plt

# a = np.random.randn(100, 5)
# soft = np.exp(a)
# soft = soft/soft.sum(axis=1, keepdims=True)
# print(soft.sum())

def sigmoid(a):
    return 1 / (1+np.exp(-a))

def forward(X, W1, b1, W2, b2):
    Z1 = sigmoid(W1.dot(X) + b1)
    out = np.exp(W2.dot(Z1) + b2)
    out = out/out.sum(axis=1, keepdims= True)
    return out

def classification(T, Y):
    count= 0
    for i in range(len(Y)):
        if Y[i] == T[i]:
            count+=1
    return count/len(Y)




N = 500
X1 = np.random.randn(N, 2) + np.array([0, -2])
X2 = np.random.randn(N, 2) + np.array([2, 2])
X3 = np.random.randn(N, 2) + np.array([-2, 2])
X = np.vstack((X1, X2, X3))
X = X.T
print(X.shape)
Y = np.array([0] * N + [1] * N + [2] * N).T
plt.scatter(X[0,:], X[1,:], c=Y, s=100, alpha=0.5)
plt.show()

#dimesnions is 2
D = 2
#hidden layer perceptrons 3
M = 3
#outputlayer is 3
K = 3

W1 = np.random.randn(M, D)
b1 = np.random.randn(M,1)
W2 = np.random.randn(K, M)
b2 = np.random.randn(K,1)

Yhat = forward(X, W1, b1, W2, b2)
print(np.shape(Yhat))
Yhat = np.argmax(Yhat, axis = 0)
print(np.shape(Yhat))
print("Classification rate for randomly chosen weights:",classification(Y, Yhat))