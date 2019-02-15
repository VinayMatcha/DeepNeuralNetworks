import numpy as np
import pandas as pd

def get_data():
    df = pd.read_csv('ecommerce_data.csv')
    data = df.values
    X = data[:, :-1]
    Y = data[:, -1]
    X[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    X[:, 2] = (X[:, 2] - X[:, 2].mean()) / X[:, 2].std()
    N, D = X.shape
    finalX = np.zeros((N, D+3))
    finalX[:,0:D-1] = X[:, 0:D-1]
    for i in range(N):
        t = int(X[i, D-1])
        finalX[i, t+D-1] = 1
    Z = np.zeros((N, 4))
    Z[np.arange(N), X[:,D-1].astype(np.int32)] = 1
    #finalX:,-4] = Z
    print()
    assert(np.abs(finalX[:,-4:]-Z).sum() < 1e-5)
    return finalX, Y


#to get data for logistic regression that is for only 2 classes
def get_binary_data():
    X, Y = get_data()
    X2 = X[Y<=1]
    Y2 = Y[Y<=1]
    return X2, Y2



def get_data_check():
  df = pd.read_csv('ecommerce_data.csv')

  # just in case you're curious what's in it
  # df.head()

  # easier to work with numpy array
  data = df.values

  # shuffle it
  np.random.shuffle(data)

  # split features and labels
  X = data[:,:-1]
  Y = data[:,-1].astype(np.int32)

  # one-hot encode the categorical data
  # create a new matrix X2 with the correct number of columns
  N, D = X.shape
  X2 = np.zeros((N, D+3))
  X2[:,0:(D-1)] = X[:,0:(D-1)] # non-categorical

  # one-hot
  for n in range(N):
      t = int(X[n,D-1])
      X2[n,t+D-1] = 1

  # method 2
  # Z = np.zeros((N, 4))
  # Z[np.arange(N), X[:,D-1].astype(np.int32)] = 1
  # # assign: X2[:,-4:] = Z
  # assert(np.abs(X2[:,-4:] - Z).sum() < 1e-10)

  # assign X2 back to X, since we don't need original anymore
  X = X2

  # split train and test
  Xtrain = X[:-100]
  Ytrain = Y[:-100]
  Xtest = X[-100:]
  Ytest = Y[-100:]

  # normalize columns 1 and 2
  for i in (1, 2):
    m = Xtrain[:,i].mean()
    s = Xtrain[:,i].std()
    Xtrain[:,i] = (Xtrain[:,i] - m) / s
    Xtest[:,i] = (Xtest[:,i] - m) / s

  return Xtrain, Ytrain, Xtest, Ytest