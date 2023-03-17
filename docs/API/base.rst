
PyTreeClass Public API
========================

.. currentmodule:: pytreeclass 

.. autosummary::
    treeclass
    is_treeclass
    is_tree_equal
    field
    fields
    tree_diagram
    tree_mermaid
    tree_repr
    tree_str
    tree_summary
    is_nondiff
    is_frozen
    unfreeze
    ImmutableWrapper
    bcmap
    tree_indexer
    register_pytree_node_trace
    tree_map_with_trace
    tree_leaves_with_trace
    tree_flatten_with_trace


`treeclass` API
---------------------

.. currentmodule:: pytreeclass 

.. autofunction:: treeclass 
.. autofunction:: is_treeclass 
.. autofunction:: is_tree_equal 
.. autofunction:: field 
.. autofunction:: fields

Pretty Printing API
-------------------

.. currentmodule:: pytreeclass 

.. autofunction:: tree_diagram
.. autofunction:: tree_mermaid 
.. autofunction:: tree_repr 
.. autofunction:: tree_str
.. autofunction:: tree_summary

Wrapping and freezing API
-------------------

.. currentmodule:: pytreeclass 

.. autofunction:: is_nondiff
.. autofunction:: is_frozen
.. autofunction:: freeze
.. autofunction:: unfreeze
.. autofunction:: FrozenWrapper
.. autofunction:: ImmutableWrapper