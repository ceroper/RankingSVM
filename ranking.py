"""
Implementation of pairwise ranking using scikit-learn LinearSVC

Reference: "Large Margin Rank Boundaries for Ordinal Regression", R. Herbrich,
    T. Graepel, K. Obermayer.

Authors: Fabian Pedregosa <fabian@fseoane.net>
         Alexandre Gramfort <alexandre.gramfort@inria.fr>
         Caroline Roper <https://www.linkedin.com/in/caroline-roper-2b623628/>
"""

import itertools
import numpy as np

from sklearn import svm, linear_model
from sklearn.model_selection import KFold


def transform_pairwise(X, y):
    """Transforms data into pairs with balanced labels for ranking

    Transforms a n-class ranking problem into a two-class classification
    problem. Subclasses implementing particular strategies for choosing
    pairs should override this method.

    In this method, all pairs are choosen, except for those that have the
    same target value. The output is an array of balanced classes, i.e.
    there are the same number of -1 as +1

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The data
    y : array, shape (n_samples,) or (n_samples, 2)
        Target labels. If it's a 2D array, the second column represents
        the grouping of samples, i.e., samples with different groups will
        not be considered.

    Returns
    -------
    X_trans : array, shape (k, n_feaures)
        Data as pairs
    y_trans : array, shape (k,)
        Output class labels, where classes have values {-1, +1}
    """
    X_new = []
    y_new = []
    y = np.asarray(y)
    if y.ndim == 1:
        y = np.c_[y, np.ones(y.shape[0])]
    comb = itertools.combinations(range(X.shape[0]), 2)
    for k, (i, j) in enumerate(comb):
        if y[i, 0] == y[j, 0] or y[i, 1] != y[j, 1]:
            # skip if same target or different group
            continue
        X_new.append(X[i] - X[j])
        y_new.append(np.sign(y[i, 0] - y[j, 0]))
        # output balanced classes
        if y_new[-1] != (-1) ** k:
            y_new[-1] = - y_new[-1]
            X_new[-1] = - X_new[-1]
    return np.asarray(X_new), np.asarray(y_new).ravel()


class RankSVM(svm.LinearSVC):
    """Performs pairwise ranking with an underlying LinearSVC model

    Input should be a n-class ranking problem, this object will convert it
    into a two-class classification problem, a setting known as
    `pairwise ranking`.

    See object :ref:`svm.LinearSVC` for a full description of parameters.
    """

    def fit(self, X, y):
        """
        Fit a pairwise ranking model.

        Parameters
        ----------
        X : array, shape (n_samples, n_features + grouping column)
        y : array, shape (n_samples,) or (n_samples, 2)

        Returns
        -------
        self
        """
        X_data = X[:, :-1]
        X_trans, y_trans = transform_pairwise(X_data, y)
        super(RankSVM, self).fit(X_trans, y_trans)
        return self

    def predict_group(self, X):
        """
        Predict an ordering on X. For a list of n samples, this method
        returns a list from 0 to n-1 with the relative order of the rows of X.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)

        Returns
        -------
        ord : array, shape (n_samples,)
            Returns a list of integers representing the relative order of
            the rows in X.
        """
        if hasattr(self, 'coef_'):
            return np.argsort(np.dot(X, self.coef_.T).ravel())
        else:
            raise ValueError("Must call fit() prior to predict()")
        
    def predict(self, X):
        """
        Parameters
        ----------
        X: 2D numpy array, shape (n_samples, n_features + 1), last column is group id, rest are features

        Returns
        -------
        y_pred: numpy array, length n_samples, the model predictions
        """
        if not hasattr(self, 'coef_'):
            raise ValueError("Must call fit() prior to predict()")
    
        # Split features and group ids
        group_ids = X[:, -1]
        X_data = X[:, :-1]

        # Map each group to the indices belonging to it
        from collections import defaultdict
        group_to_indices = defaultdict(list)
        for idx, gid in enumerate(group_ids):
            group_to_indices[gid].append(idx)

        y_pred = np.empty(len(X_data), dtype=object)  # Use object dtype to allow storing ints or other types

        for gid, indices in group_to_indices.items():
            # Extract features for this group
            x_group = X_data[indices, :]
            # Fill missing values if needed (e.g., with 0)
            x_group = np.nan_to_num(x_group)  # replaces NaN with 0
            x_group = x_group.astype(np.int32)

            # Predict for the group
            preds_group = self.predict_group(x_group)

            # Assign predictions to correct positions
            y_pred[indices] = preds_group


        return y_pred

    def score_group(self, X, y):
        """
        Because we transformed into a pairwise problem, chance level is at 0.5
        """
        X_trans, y_trans = transform_pairwise(X, y)
        return np.mean(super(RankSVM, self).predict(X_trans) == y_trans)