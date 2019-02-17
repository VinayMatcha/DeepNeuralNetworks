from basics.process import get_data_check
from sklearn.neural_network import MLPClassifier
def main():
    X_train, Y_train, X_test, Y_test = get_data_check()
    model = MLPClassifier(max_iter=2000)
    model.fit(X_train, Y_train)
    print(model.score(X_train, Y_train))
    print(model.score(X_test, Y_test))


if __name__ == '__main__':
    main()