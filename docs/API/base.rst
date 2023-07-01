
Public API
========================

.. currentmodule:: pytreeclass 

.. autosummary::
    TreeClass
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
    pp_dispatcher
    is_nondiff
    freeze
    tree_freeze
    is_frozen
    unfreeze
    tree_unfreeze
    bcmap
    Partial
    AtIndexer
    RegexKey
    BaseKey
    tree_map_with_trace
    tree_leaves_with_trace
    tree_flatten_with_trace


core API
---------------------

.. currentmodule:: pytreeclass 

.. autoclass:: TreeClass 
    :members: at
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
.. autofunction:: pp_dispatcher

Masking/Unmasking API
-------------------------

.. currentmodule:: pytreeclass 

.. autofunction:: is_nondiff
.. autofunction:: freeze
.. autofunction:: unfreeze
.. autofunction:: is_frozen
.. autofunction:: tree_mask
.. autofunction:: tree_unmask


Advanced API
------------
.. currentmodule:: pytreeclass

.. autofunction:: bcmap
.. autoclass:: Partial
.. autoclass:: AtIndexer
    :members:
        get,
        set,
        apply,
        scan,
.. autoclass:: RegexKey
    :members:
        __eq__
.. autoclass:: BaseKey
    :members:
        __eq__
.. autofunction:: tree_map_with_trace
.. autofunction:: tree_leaves_with_trace
.. autofunction:: tree_flatten_with_trace
