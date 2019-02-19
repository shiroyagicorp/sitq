================================
SITQ - Learning to Hash for MIPS
================================

SITQ is a fast algorithm for approximate Maximum Inner Product Search (MIPS).
It can find items which are likely to maximize inner product against a query in sublinear time.

Benchemark
==========

Recommendation is one of fields where SITQ can be used.
Experiments were conducted with `MovieLens 100K Dataset` and `MovieLens 20M Dataset`.

ALS in `benfred/implicit <https://github.com/benfred/implicit>`_ is used to learn vectors of items and users, where the score of \(user, item\) pair is computed through inner product of those vectors.
`Precision@10` is the ratio of correct recommendations against test dataset.
`Fetched items` are items against which inner product are computed.
Hashing algorithms are more preferable as average and standard deviation of `fetched items` are smaller.

ml-100k
-------

.. csv-table:: Signature length: 4, Minimum fetched items: 20
    :header: "Name", "Precision\\@10", "Fetched items. Avg", "Fetched items. Std"

    "**SITQ**", 0.202, 105.2, 76.6
    "Simple-LSH", 0.182, 496.2, 441.2
    "ITQ", 0.199, 131.3, 93.7
    "LSH", 0.156, 161.9, 94.4
    "brute force", 0.242, \(1680\)

ml-20m
------

.. csv-table:: Signature length: 8, Minimum fetched items: 20
    :header: "Name", "Precision\\@10", "Fetched items. Avg", "Fetched items. Std"

    "**SITQ**", 0.112, 96.1, 151.1
    "Simple-LSH", 0.122, 2158.2, 5246.6
    "ITQ", 0.090, 111.0, 332.9
    "LSH", 0.069, 531.3, 912.2
    "brute force", 0.151, \(26671\)

Algorithm
=========

SITQ is an algorithm which combines Simple-LSH [1]_ and ITQ [2]_.

Simple-LSH utilizes ordinary LSH which is for cosine similarity.
In order to use LSH for MIPS, it converts a vector before computing its signature.

LSH computes signatures through transformation matrix which is fixed.
ITQ learns transformation matrix from item vectors for better hashing.

SITQ converts vectors by means of Simple-LSH, and learns transformation matrix through ITQ.

Example
=======

Install
-------

::

    pip install sitq

Get Signature
-------------

.. code-block:: python

    import numpy as np

    from sitq import Sitq


    # Create sample dataset
    items = np.random.rand(10000, 50)
    query = np.random.rand(50)

    sitq = Sitq(signature_size=8)

    # Learn transformation matrix
    sitq.fit(items)

    # Get signatures for items
    item_sigs = sitq.get_item_signatures(items)

    # Get signature for query
    query_sig = sitq.get_query_signatures([query])[0]

Retrieve items
--------------

.. code-block:: python

    import numpy as np

    from sitq import Mips


    # Create sample dataset
    items = np.random.rand(10000, 50)
    query = np.random.rand(50)

    mips = Mips(signature_size=8)

    # Learn lookup table and parameters for search
    mips.fit(items)

    # Find items which are likely to maximize inner product against query
    item_indexes, scores = mips.search(query, limit=10, distance=1)

References
----------

.. [1] Neyshabur, Behnam, and Nathan Srebro. "On symmetric and asymmetric LSHs for inner product search." arXiv preprint arXiv:1410.5518 (2014).
.. [2] Gong, Yunchao, et al. "Iterative quantization: A procrustean approach to learning binary codes for large-scale image retrieval." IEEE Transactions on Pattern Analysis and Machine Intelligence 35.12 (2013): 2916-2929.
