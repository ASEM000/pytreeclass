
Public API
========================

.. currentmodule:: pytreeclass 

.. autosummary::
    treeclass
    is_treeclass
    is_tree_equal
    field
    fields
    tree_diagram
    tree_indent
    tree_mermaid
    tree_repr
    tree_str
    tree_summary
    tree_repr_with_trace
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


core API
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
.. autofunction:: tree_indent
.. autofunction:: tree_mermaid 
.. autofunction:: tree_repr 
.. autofunction:: tree_str
.. autofunction:: tree_summary
.. autofunction:: tree_repr_with_trace

Wrapping and freezing API
-------------------------

.. currentmodule:: pytreeclass 

.. autofunction:: is_nondiff
.. autofunction:: is_frozen
.. autofunction:: freeze
.. autofunction:: unfreeze
.. autoclass:: ImmutableWrapper


Advanced API
------------
.. currentmodule:: pytreeclass

.. autofunction:: bcmap
.. autofunction:: tree_indexer
.. autofunction:: register_pytree_node_trace
.. autofunction:: tree_map_with_trace
.. autofunction:: tree_leaves_with_trace
.. autofunction:: tree_flatten_with_trace