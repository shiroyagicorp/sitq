from collections import defaultdict, namedtuple
from copy import copy
from itertools import combinations

import numpy as np

from ._sitq import Sitq

Item = namedtuple('Item', ['name', 'vector'])


class Mips:
    def __init__(self, signature_size):
        """
        Parameters
        ----------
        signature_size: int
            The number of bits of a signature.
        """

        self._sitq = Sitq(signature_size=signature_size)

    def fit(self, items, iteration=50):
        """
        Learn lookup table and parameters for search.

        Parameters
        ----------
        items: array_like or dict
            Training data. If it is array_like, shape of (n_items, n_features)
            where n_items is the number of items and n_features is the number 
            of features. If it is dict, the key is the name of the item and the
            value is array_like with shape of (n_features).
        iteration: int, optional
            The number of iteration for learning ITQ encoder.

        Returns
        -------
        self: object
            Returns the instance itself.
        """
        _items = []
        if isinstance(items, dict):
            for name, vector in items.items():
                _items.append(Item(name, vector))
        else:
            for name, vector in enumerate(items):
                _items.append(Item(name, vector))

        item_vectors = [item.vector for item in _items]
        self._sitq.fit(item_vectors, iteration=iteration)
        sigs = self._sitq.get_item_signatures(item_vectors)

        self._table = defaultdict(list)
        for sig, item in zip(sigs, _items):
            _sig = tuple(sig)
            self._table[_sig].append(item)

        return self

    def search(self, query, limit=None, max_distance=0, sort=True):
        """
        Find items which are likely to maximize inner product against query.

        Parameters
        ----------
        query: array_like, shape(n_features)
            Vector for query.
        limit: int or None, optional
            The maximum number of items to be returned.
        max_distance: int, optional
            The number of bits by which query and signature can differ.
        sort: bool, optional
            If true, then the returned `item_names` are sorted in descending
            order according to these `scores`.

        Returns
        -------
        item_names: ndarray
            Names of items. Indexes are used as names when array_like was used
            for `fit()`.
        scores: ndarray
            Inner prodects of items.
        """
        query_sig = self._sitq.get_query_signatures([query])[0]

        items = []
        max_distance = np.clip(max_distance, 0, self._sitq._signature_size)
        for i in range(max_distance + 1):
            for mutation_indexes in combinations(range(max_distance), i):
                mutated_sig = copy(query_sig)
                for idx in mutation_indexes:
                    mutated_sig[idx] = bool((mutated_sig[idx] + 1) % 2)
                _sig = tuple(mutated_sig)
                items += self._table[_sig]

        items = np.array(items)
        item_vectors = np.array([item[1] for item in items])
        scores = item_vectors.dot(query)
        if limit is not None:
            limit = np.clip(limit, 0, len(scores))
            idxs = np.argpartition(scores, -1 * limit)[-1 * limit:]
            items = items[idxs]
            scores = scores[idxs]

        if sort:
            idxs = np.argsort(scores)[::-1]
            items = items[idxs]
            scores = scores[idxs]

        item_names = np.array([item[0] for item in items])

        return item_names, scores
