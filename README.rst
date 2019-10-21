funlib.match
===========

.. image:: https://travis-ci.com/funkelab/funlib.match.svg?branch=master
  :target: https://travis-ci.com/funkelab/funlib.match

Currently only supports matching graph *G* to graph *S*, where *S* is an arborescence,
and *G* is an overcomplete graph and contains a subgraph isomorphic to *S*.

Installation
============

Most of the requirements can be installed through pip, except pylp which must
be installed with *conda install -c funkey pylp* and rtree which must be
installed through *conda install rtree* to properly install the *C* spatial
libraries.

Implementation
==============

The graph matching implementation is adapted from:

https://hal.archives-ouvertes.fr/hal-00726076/document
Pierre Le Bodic, Pierre Héroux, Sébastien Adam, Yves Lecourtier. An integer linear
program for substitution-tolerant subgraph isomorphism and its use for symbol
spotting in technical drawings.
Pattern Recognition, Elsevier, 2012, 45 (12), pp.4214-4224. ffhal-00726076

Indicators
==========

For every vertex *i* in *V(G)* and vertex *j* in *V(S) Union {None}*, *x_ij = 1* if *i* matches to *j*
For every edge *ij* in E(*S*) and edge *kl* in *V(G)*, *y_ij_kl = 1* if *ij* matches to *kl*

Constraints
===========

1) *S* to *G* vertexs mapping:
    Every vertex in *V(S)* must map to exactly 1 vertex in *V(G)*
2) *S* to *G* edge mapping:
    Every edge in E(*S*) must map to at least 1 edge in E(*G*)
3) *G* to *S* vertexs mapping:
    Every vertex in *V(G)* must map to exctaly 1 vertex in *V(S) Union {None}*
4) In edge mapping:
    For every vertex *g* in *V(G)* that maps to a vertex *s* in *V(S)*, any edge
    in E(*S*) that targets *s* must match to an edge in E(*G*) that targets *g*
5) Out edge mapping:
    For every vertex *g* in *V(G)* that maps to a vertex in *V(S)*, any edge
    in E(*S*) that originates from *s* must match to an edge in E(*G*) that originates
    from *g*
6) Balance (Added constraint):
    For every vetex *g* in *V(G)* that maps to a vertex *s* in *V(S) Union {None}*,
    every edge that targets *s*, must be matched to *n+1* edges targeting *g* and
    *n* edges originating from *g*. similarly every edge originating from *s* must
    be matched to *n+1* edges originating from *g*, and *n* edges targeting *g*.
    Finally, any edge neither targeting nor originating from *s*, must be matched
    to *n* edges originating from *g* and *n* edges targeting *g*.
7) Degree (Added constraint):
    For every vertex *g* in *V(G)* that maps to a vertex *s* in *V(S) Union {None}*,
    the degree of *g* (not counting unmatched edges) must equal the degree of
    *s* if *s* is not None, else 2.

Constraints 6 and 7 were added to enforce topologically accurate "chains".
Shese are cases where 1 edge in *S* is very long, and is best represented
by a series of edges in *G*.
Constraint 6 is redundant to 4 and 5 on matched vertices, but on unmatched
vertices, it enforces a 1-1 mapping from matched targeting edges to matched
originating edges.
Constraint 7 eliminates the possibility of crossovers where one unmatched
node has multiple chains passing through it. Note that this constraint in
combination with 4 and 5, ensure that the *n* in 6 is always 0.
