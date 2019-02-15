import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from basics.process import get_data
from basics.logistic_ecommerce import Classifucation, y2indicator, predict, cross_entropy, softmax

def forward(X, W1, b1, W2, b2):
    Z = np.tanh(X.dot(W1) + b1)
    return softmax(Z.dot(W2)+b2), Z


def main():

    X, Y = get_data()
    X, Y = shuffle(X, Y)
    Y = Y.astype(np.int32)

    N, D = X.shape
    M = 5
    K = len(set(Y))

    X_train = X[:-100]
    Y_train = Y[:-100]
    X_test = X[-100:]
    Y_test = Y[-100:]
    Y_trains = y2indicator(Y_train, K)
    Y_tests = y2indicator(Y_test, K)

    W1 = np.random.randn(D, M)
    b1 = np.random.randn(M)
    W2 = np.random.randn(M, K)
    b2 = np.random.randn(K)


    train_costs = []
    test_costs = []
    lr = 0.001
    for i in range(10000):
        pYtrain, Ztrain = forward(X_train, W1, b1, W2, b2)
        pYtest, Ztest = forward(X_test, W1, b1, W2, b2)

        ctrain = cross_entropy(Y_trains, pYtrain)
        ctest = cross_entropy(Y_tests, pYtest)
        train_costs.append(ctrain)
        train_costs.append(ctest)

        W2 -= lr * Ztrain.T.dot(pYtrain - Y_trains)
        b2 -= lr * (pYtrain - Y_trains).sum(axis=0)
        dZ = (pYtrain - Y_trains).dot(W2.T) * (1 - Ztrain * Ztrain)
        W1 -= lr * X_train.T.dot(dZ)
        b1 -= lr * dZ.sum(axis=0)

        if i % 1000 == 0:
            print(i, "th itearation is costs are ", ctrain, ctest)

    print("classifcation rate for test and train is respectievly", Classifucation(Y_train, predict(pYtrain)), " and ", Classifucation(Y_test, predict(pYtest)))

    legend1, = plt.plot(train_costs, label = 'train_costs')
    legend2, = plt.plot(test_costs, label = 'test_costs')
    plt.legend([legend1, legend2])
    plt.show()
    plt.savefig('images/NeuralNetEcommerceCosts')


if __name__ == '__main__':
    main()