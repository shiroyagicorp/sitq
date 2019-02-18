import numpy as np

from sitq import Sitq

def test_sitq():
    items = np.random.randn(10000, 50)
    queries = np.random.randn(100, 50)
    sitq = Sitq(signature_size=4).fit(items)
    assert sitq.get_item_signatures(items).shape == (10000, 4)
    assert sitq.get_query_signatures(queries).shape == (100, 4)
