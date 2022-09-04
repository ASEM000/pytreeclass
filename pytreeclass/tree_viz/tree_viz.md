# tree_viz package

This subpackage is composed of:

- Table drawing functions `_hbox`, `_vbox`, `_table` .
- Pretty printer `_format_node_repr`/ `_format_node_str` for
  - Python containers (`list`/`tuple`/`dict`/`set`).
  - `jax.numpy.ndarray`.
  - Functions.
  - `treeclass` data structrue.
- Treeclass to `mermaid.live` flow chart converter.
- Summary for `treeclass` data structure.
