import numpy as np

from sitq import Mips


def _brute_force(items, query):
    return np.argsort(items.dot(query))[::-1]


def test_mips():
    items = np.random.randn(10000, 50)
    query = np.random.randn(50)
    mips = Mips(signature_size=4).fit(items)

    item_idxs, scores = mips.search(
        query, limit=None, distance=4, sort=False)
    assert len(item_idxs) == len(scores) == len(items)

    item_idxs, scores = mips.search(
        query, limit=10, distance=2, sort=False)
    assert len(item_idxs) == len(scores) == 10

    _lens = [len(_) for _ in mips._vector_table.values()]
    item_idxs, scores = mips.search(
        query, limit=None, distance=0, sort=False)
    assert min(_lens) <= len(item_idxs) == len(scores) <= max(_lens)

    item_idxs, scores = mips.search(
        query, limit=None, distance=1, sort=False)
    assert min(_lens) < len(item_idxs) == len(scores) < len(items)

    item_idxs, scores = mips.search(
        query, limit=10, distance=10, sort=True)
    assert len(item_idxs) == len(scores) == 10
    assert items[item_idxs[0]].dot(query) > items[item_idxs[-1]].dot(query)

    item_idxs, scores = mips.search(
        query, limit=100000, distance=10, sort=False)
    assert len(item_idxs) == len(scores) == len(items)

    item_idxs, scores = mips.search(
        query, limit=None, distance=1, sort=True)
    assert scores[0] > scores[1]


def test_few_items():
    items = np.random.rand(4, 10)
    queries = np.random.randn(100, 10)
    mips = Mips(signature_size=4).fit(items)

    assert min(len(mips.search(query)[0]) for query in queries) == 0
    assert min(len(mips.search(query, limit=len(items), require_items=True)[0]) for query in queries) == len(items)
    assert min(len(mips.search(query, limit=2, require_items=True)[0]) for query in queries) == 2
    assert min(len(mips.search(query, limit=2, require_items=False)[0]) for query in queries) < 2


def test_precision():
    items = np.random.randn(10000, 50)
    queries = np.random.randn(100, 50)

    mips = Mips(signature_size=4).fit(items)

    acc = 0
    for query in queries:
        correct_idxs = _brute_force(items, query)
        idxs, _ = mips.search(query, limit=1, distance=0)
        if correct_idxs[0] in idxs:
            acc += 1

    precision = acc / queries.shape[0]
    assert precision > 1 / 2 ** 4
