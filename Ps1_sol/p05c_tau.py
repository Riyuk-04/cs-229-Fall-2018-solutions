import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval,y_eval = util.load_dataset(valid_path, add_intercept=True)
    x_test,y_test = util.load_dataset(test_path, add_intercept=True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    MSE_array = []
    for i in range(np.shape(tau_values)[0]):
        clf = LocallyWeightedLinearRegression(tau_values[i])
        clf.fit(x_train, y_train)
        labels = clf.predict(x_eval)
        MSE = ((y_eval - labels).dot(y_eval - labels))/y_eval.shape[0]
        MSE_array.append(MSE)
        print("Tau = ",tau_values[i],"  MSE = ",MSE)
        #plt.plot(x_train,y_train,'bx')
        #plt.plot(x_eval,labels,'ro')
        #plt.show()
    ind = np.argmin(MSE_array)
    # Fit a LWR model with the best tau value
    model = LocallyWeightedLinearRegression(tau_values[ind])
    model.fit(x_train, y_train)
    # Run on the test set to get the MSE value
    labels = model.predict(x_test)
    MSE = ((y_test - labels).dot(y_test - labels))/y_test.shape[0]
    print("Lowest MSE = ",MSE_array[ind])
    print("MSE_Test = ",MSE)
    # Save predictions to pred_path
    np.savetxt(pred_path,labels)
    # Plot data
    plt.plot(x_train,y_train,'bx')
    plt.plot(x_test,labels,'ro')
    plt.show()
    # *** END CODE HERE ***

