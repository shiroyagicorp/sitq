import pqkmeans
import numpy as np


class Sitq:
    def __init__(self, signature_size):
        """
        Parameters
        ----------
        signature_size: int
            The number of bits of a signature.
        """

        self._signature_size = signature_size

    def _p(self, X):
        Xl = X / self._max_norm
        Xr = np.sqrt(
            np.clip(1 - np.linalg.norm(Xl, axis=1) ** 2, 0.0, 1.0)
        )[:, np.newaxis]
        return np.concatenate((Xl, Xr), axis=1)

    def fit(self, items, iteration=50):
        """
        Learn parameters for calculating signatures.

        Parameters
        ----------
        items: array_like, shape(n_items, n_features)
            Training data, where n_items is the number of items and n_features is the number of features.
        iteration: int, optional
            The number of iteration for learning ITQ encoder.

        Returns
        -------
        self: object
            Returns the instance itself.
        """
        self._max_norm = np.max(np.linalg.norm(items, axis=1))
        X = self._p(items)

        encoder = pqkmeans.encoder.ITQEncoder(iteration=iteration,
                                              num_bit=self._signature_size)
        encoder.fit(X)

        self._pca_mean = encoder.trained_encoder.pca.mean_
        self._itq_pr = encoder.trained_encoder.pca.components_.T.dot(
            encoder.trained_encoder.R)

        return self

    def _get_signatures(self, X):
        return (X - self._pca_mean).dot(self._itq_pr) >= 0

    def get_item_signatures(self, items):
        """
        Get signatures for items.

        Parameters
        ----------
        items: array_like, shape(n_items, n_features)

        Returns
        -------
        signatures: ndarray, shape(n_items, signature_size)
        """
        X = self._p(items)
        return self._get_signatures(X)

    def get_query_signatures(self, queries):
        """
        Get signatures for queries.

        Parameters
        ----------
        queries: array_like, shape(n_queries, n_features)

        Returns
        -------
        signatures: ndarray, shape(n_queries, signature_size)
        """
        X = np.concatenate((
            queries / np.linalg.norm(queries, axis=1)[:, np.newaxis],
            np.zeros([len(queries), 1])
        ), axis=1)
        return self._get_signatures(X)
