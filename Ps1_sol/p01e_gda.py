import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_eval,y_eval = util.load_dataset(eval_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a GDA classifier
    clf = GDA()
    clf.fit(x_train, y_train)
    # Plot decision boundary on validation set
    util.plot(x_train,y_train,clf.theta)
    util.plot(x_eval,y_eval,clf.theta)
    # Use np.savetxt to save outputs from validation set to pred_path
    np.savetxt(pred_path,clf.predict(x_eval))
    # *** END CODE HERE ***

class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma
        self.theta = np.zeros(x.shape[1] + 1)
        mu_0 = np.zeros(x.shape[1])
        mu_1 = np.zeros(x.shape[1])
        sigma = np.zeros(shape = (x.shape[1],x.shape[1]))
        count_0 = 0
        count_1 = 0

        for i in range(x.shape[0]):
            a = x[i]
            a = np.reshape(a,(x.shape[1],1))
            sigma += np.dot(a,np.transpose(a))
            if y[i] == 1:
                count_1 += 1
                mu_1 += x[i]
            else :
                count_0 += 1
                mu_0 += x[i]

        phi = count_1/(count_1 + count_0)
        mu_0 = mu_0/x.shape[0]
        mu_1 = mu_1/x.shape[0]
        mu_0 = np.reshape(mu_0,(mu_0.shape[0],1))
        mu_1 = np.reshape(mu_1,(mu_1.shape[0],1))
        sigma = sigma/x.shape[0]

        # Write theta in terms of the parameters
        sigma_inv = np.linalg.inv(sigma)
        theta_0 = np.log(phi/(1 - phi)) + 0.5*(np.dot(np.transpose(mu_0),np.dot(sigma_inv,mu_0)) - np.dot(np.transpose(mu_1),np.dot(sigma_inv,mu_1)))
        theta_array = np.dot(sigma_inv,(mu_1 - mu_0))
        
        for i in range(x.shape[1] + 1):
            if i == 0:
                self.theta[i] = theta_0
            else:
                self.theta[i] = theta_array[i-1]    
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        labels = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            if np.dot(np.transpose(x[i]),self.theta) > 0:
                labels[i] = 1
        return labels
        # *** END CODE HERE
