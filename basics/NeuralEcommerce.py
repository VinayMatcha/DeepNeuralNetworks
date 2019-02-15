import numpy as np
from basics.process import get_data


def softmax(Z):
    expz = np.exp(Z)
    return expz/expz.sum(axis =1, keepdims=True)

def forward(X, W1, b1, W2, b2):
    Z = np.tanh(X.dot(W1) + b1)
    return softmax(Z.dot(W2)+b2)

def classification_rate(y, y_hat):
    # return np.mean(y==y_hat)
    count = 0
    for i in range(len(y)):
        # print(y[i].shape)
        if(y[i] == y_hat[i]):
            count += 1
    return count/len(y)

X, Y = get_data()
N, D = X.shape
M = 5
K = len(set(Y))

W1 = np.random.randn(D, M)
b1 = np.random.randn(M)
W2 = np.random.randn(M, K)
b2 = np.random.randn(K)

y_hat = forward(X, W1, b1, W2, b2)
y_hat1 = np.argmax(y_hat, axis=1)
eff = classification_rate(Y, y_hat1)
print(eff)





