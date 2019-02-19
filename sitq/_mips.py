from collections import defaultdict, namedtuple
from itertools import combinations

import numpy as np

from ._sitq import Sitq


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
        Item = namedtuple('Item', ['name', 'vector'])

        _items = []
        if isinstance(items, dict):
            for name, vector in items.items():
                _items.append(Item(name, vector))
        else:
            for name, vector in enumerate(items):
                _items.append(Item(name, vector))

        item_vectors = np.array([item.vector for item in _items])
        self._sitq.fit(item_vectors, iteration=iteration)
        sigs = self._sitq.get_item_signatures(item_vectors)

        self._vector_table = defaultdict(list)
        self._name_table = defaultdict(list)
        for sig, item in zip(sigs, _items):
            _sig = tuple(sig)
            self._vector_table[_sig].append(item.vector)
            self._name_table[_sig].append(item.name)
        self._vector_table = defaultdict(
            lambda: np.empty((0, item_vectors.shape[1]), dtype=float),
            {k: np.array(v) for k, v in self._vector_table.items()}
        )

        return self

    def search(self, query, limit=None, distance=0, require_items=False, sort=True):
        """
        Find items which are likely to maximize inner product against query.

        Parameters
        ----------
        query: array_like, shape(n_features)
            Vector for query.
        limit: int or None, optional
            The maximum number of items to be returned. It must be positive.
        distance: int, optional
            The number of bits by which query and signature can differ. 
            Maximum number of different bits may be over this number if 
            `require_items` is set to true.
        require_items: bool, optional
            If true, then the number of returned items must be `limit`.
            `distance` is ignored if the number of fetched items was
            insufficient.
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

        item_vectors = []
        item_names = []
        distance = np.clip(distance, 0, self._sitq._signature_size)
        for i in range(self._sitq._signature_size + 1):
            for mutation_indexes in combinations(range(self._sitq._signature_size), i):
                mutated_sig = query_sig.copy()
                for idx in mutation_indexes:
                    mutated_sig[idx] = not mutated_sig[idx]
                _sig = tuple(mutated_sig)
                item_vectors.append(self._vector_table[_sig])
                item_names += self._name_table[_sig]
            if (not require_items and i == distance) or \
                    (require_items and limit is not None and len(item_names) >= limit):
                break
        item_vectors = np.vstack(item_vectors)

        scores = item_vectors.dot(query) if item_vectors.size > 0 else np.empty(0)
        if limit is not None:
            limit = np.clip(limit, 0, len(scores))
            idxs = np.argpartition(scores, -1 * limit)[-1 * limit:]
        else:
            idxs = np.arange(len(item_vectors))

        if sort:
            idxs = idxs[np.argsort(scores[idxs])[::-1]]

        scores = scores[idxs]
        item_names = np.array([item_names[idx] for idx in idxs])

        return item_names, scores
