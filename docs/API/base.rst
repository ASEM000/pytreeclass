
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
    is_frozen
    unfreeze
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

Wrapping and freezing API
-------------------------

.. currentmodule:: pytreeclass 

.. autofunction:: is_nondiff
.. autofunction:: is_frozen
.. autofunction:: freeze
.. autofunction:: unfreeze


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
        __call__,
        __getitem__
.. autoclass:: RegexKey
    :members:
        __eq__
.. autoclass:: BaseKey
    :members:
        __eq__
.. autofunction:: tree_map_with_trace
.. autofunction:: tree_leaves_with_trace
.. autofunction:: tree_flatten_with_trace
