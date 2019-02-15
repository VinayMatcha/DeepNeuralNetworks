import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.special import logsumexp
from basics.process import get_data, get_data_check

def y2indicator(y, K):
    N = len(y)
    finaY = np.zeros((N, K))
    for i in range(len(y)):
        finaY[i, y[i]] = 1
    return finaY


def softmax(a):
    expa = np.exp(a - logsumexp(a))
    return expa/expa.sum(axis=1, keepdims=True)

def forward(X, W , b):
    return softmax(X.dot(W)+b)

def predict(Y_hat):
    return np.argmax(Y_hat, axis=1)

def Classifucation(Y, Y_hat):
    return np.mean(Y == Y_hat)


def classification(T, Y):
    count= 0
    for i in range(len(Y)):
        if Y[i] == T[i]:
            count+=1
    return count/len(Y)

def cross_entropy(T, py):
    return -np.mean(T*np.log(py))

def main():
    X, Y = get_data()
    X, Y = shuffle(X, Y)
    Y = Y.astype(np.int32)
    N, D = X.shape
    K = len(set(Y))


    X_train = X[:-100]
    Y_train = Y[:-100]
    print(len(Y_train))
    X_test = X[-100:]
    Y_test = Y[-100:]
    Y_trains = y2indicator(Y_train, K)
    Y_tests = y2indicator(Y_test, K)
    W = np.random.randn(D, K)
    b = b = np.zeros(K)

    train_costs = []
    test_costs = []
    lr = 0.001
    for i in range(10000):
        Y_hatTrain = forward(X_train, W, b)
        Y_hatTest = forward(X_test, W, b)
        train_costs.append(cross_entropy(Y_trains, Y_hatTrain))
        test_costs.append(cross_entropy(Y_tests, Y_hatTest))
        W -= lr * (X_train.T.dot(Y_hatTrain - Y_trains))
        b -= lr * (Y_hatTrain- Y_trains).sum(axis = 0)
        if i%1000==0:
            print(train_costs[len(train_costs)-1], test_costs[len(test_costs)-1])

    print("classifcation rate for test and train is respectievly", Classifucation(Y_train, predict(Y_hatTrain)), " and ", Classifucation(Y_test, predict(Y_hatTest)))

    legend1, = plt.plot(train_costs, label = 'train_costs')
    legend2, = plt.plot(test_costs, label = 'test_costs')
    plt.legend([legend1, legend2])
    plt.show()
    plt.savefig('images/LogisticEcommerceCosts')


if __name__ == '__main__':
    main()