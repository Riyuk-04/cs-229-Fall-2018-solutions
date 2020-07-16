import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval,y_eval = util.load_dataset(eval_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    # Plot decision boundary on top of validation set set
    util.plot(x_train,y_train,clf.theta)
    util.plot(x_eval,y_eval,clf.theta)
    # Use np.savetxt to save predictions on eval set to pred_path
    np.savetxt(pred_path,clf.predict(x_eval))
    # *** END CODE HERE ***

def hypo(x,theta):
    a = np.dot(np.transpose(x),theta)
    return 1.0/(1 + np.exp(-a))


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        Hessian = np.zeros(shape = (x.shape[1],x.shape[1]))
        self.theta = np.zeros(x.shape[1])
        norm = 1
        while (norm > self.eps):
            Del_l = np.zeros(x.shape[1])

            for i in range(x.shape[0]):
                a = x[i]
                a = np.reshape(a,(x.shape[1],1))
                hypothesis = hypo(x[i],self.theta)
                Del_l += x[i]*(hypothesis - y[i])
                Hessian += np.dot(a,np.transpose(a))*hypothesis*(1 - hypothesis)

            Del_l = Del_l*(1.0/x.shape[0])
            Hessian = Hessian*(1.0/x.shape[0])
            change = np.dot(np.linalg.inv(Hessian),Del_l)
            self.theta -= change
            norm = np.linalg.norm(change)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return 1.0/(1+np.exp(-(x.dot(self.theta))))
        # *** END CODE HERE ***
