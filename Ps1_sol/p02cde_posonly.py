import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels

    x_train, t_train = util.load_dataset(train_path, 't' ,add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, 't' ,add_intercept=True)
    clf = LogisticRegression()
    clf.fit(x_train,t_train)
    result_t = clf.predict(x_test)
    np.savetxt(pred_path_c,result_t)
    util.plot(x_test,t_test,clf.theta)

    # Make sure to save outputs to pred_path_c
    # Part (d): Train on y-labels and test on true labels

    x_train, y_train = util.load_dataset(train_path, 'y' ,add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, 't' ,add_intercept=True)
    clf2 = LogisticRegression()
    clf2.fit(x_train,y_train)
    result_y = clf2.predict(x_test)
    np.savetxt(pred_path_d,result_y)
    util.plot(x_test,t_test,clf2.theta)

    # Make sure to save outputs to pred_path_d
    # Part (e): Apply correction factor using validation set and test on true labels

    x_val, y_val = util.load_dataset(valid_path, 'y' ,add_intercept=True)
    result_y_valid = clf2.predict(x_val)
    alpha = 0
    count = 0
    for h in range(result_y_valid.shape[0]):
    	if y_val[h] == 1:
    		count += 1 
    		alpha += result_y_valid[h]

    alpha = alpha/count
    result_t_readjust = result_y/alpha
    np.savetxt(pred_path_e,result_t_readjust)
    util.plot(x_test,t_test,clf2.theta,alpha)

    # Plot and use np.savetxt to save outputs to pred_path_e
    # *** END CODER HERE


