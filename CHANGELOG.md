# Changelog

## `PyTreeClass` v0.5.1
- allow nested mutations using `.at[method](*args, **kwargs)`.
  After the change, inner methods can mutate copied new instances at any level not just the top level.
  a motivation for this is to experiment with lazy initialization scheme, where inner layers need to mutate their inner state.


  <details>

    ```python

    import pytreeclass as pytc
    import jax.random as jr
    from typing import Any
    import jax
    import jax.numpy as jnp


    @pytc.autoinit
    class LazyLinear(pytc.TreeClass):
        out_features: int

        def param(self, name: str, value: Any):
            if name not in vars(self):
                setattr(self, name, value)
            return vars(self)[name]

        def __call__(self, x: jax.Array, *, key: jr.KeyArray = jr.PRNGKey(0)):
            in_features = self.param("in_features", x.shape[-1])
            weight = self.param("weight", jnp.ones((in_features, self.out_features)))
            bias = self.param("bias", jnp.zeros((self.out_features,)))
            return x @ weight + bias


    @pytc.autoinit
    class StackedLinear(pytc.TreeClass):
        l1: LazyLinear = LazyLinear(10)
        l2: LazyLinear = LazyLinear(10)

        def call(self, x: jax.Array):
            return self.l2(self.l1(x))

    l = StackedLinear()
    print(repr(l))
    # StackedLinear(l1=LazyLinear(out_features=10), l2=LazyLinear(out_features=10))


    _, ll = l.at["call"](jnp.ones((1, 5)))
    ll
    # StackedLinear(
    #   l1=LazyLinear(
    #     out_features=10, 
    #     in_features=5, 
    #     weight=f32[5,10](μ=1.00, σ=0.00, ∈[1.00,1.00]), 
    #     bias=f32[10](μ=0.00, σ=0.00, ∈[0.00,0.00])
    #   ), 
    #   l2=LazyLinear(
    #     out_features=10, 
    #     in_features=10, 
    #     weight=f32[10,10](μ=1.00, σ=0.00, ∈[1.00,1.00]), 
    #     bias=f32[10](μ=0.00, σ=0.00, ∈[0.00,0.00])
    #   )
    # )

    ```
    </details>

## `PyTreeClass` v0.5post0

- fix `__init_subclass__`. not accepting arguments. this bug is introduced since `v0.5`


## `PyTreeClass` v0.5

## Breaking changes

#### __Auto generation of `__init__` method from type hints is decoupled from `TreeClass`__

__Alternatives__

Use:

1) _Preferably_ decorate with `pytreeclass.autoinit` with `pytreeclass.field` as field specifier. as `pytreeclass.field` has more features (e.g. `callbacks`, multiple argument kind selection) and the init generation is cached compared to `dataclasses`.
2) decorate with `dataclasses.dataclass` with `dataclasses.field` as field specifier. however :
   1) _Must_ set `fronzen=False` because the `__setattr__`, `__delattr__` is handled by `TreeClass`
   2) _Optionally_ `repr=False` to be handled by `TreeClass`
   3) _Optionally_ `eq=hash=False` as it is handled by `TreeClass`

<div align="center">

<table>
<tr>

<td>

### Before

```python
import jax.tree_util as jtu
import pytreeclass as pytc
import dataclasses as dc

class Tree(pytc.TreeClass):
    a: int = 1

jtu.tree_leaves(Tree())
# [1]

```

</td>

<td>

### After
Equivalent behavior when decorating with either: 

1) `@pytreeclass.autoinit`
2) `@dataclasses.dataclass` 

```python
import jax.tree_util as jtu
import pytreeclass as pytc

@pytc.autoinit
class Tree(pytc.TreeClass):
    a: int = 1

jtu.tree_leaves(Tree())
# [1]

```

</td>

<tr>

</table>

</div>


This change aims to fix the ambiguity of using the `dataclass` mental model in the following siutations:

1) subclassing. previously, using `TreeClass` as a base class is equivalent to decorating the class with `dataclasses.dataclass`, however this is a bit challenging to understand as demonstrated in the next example:

    ``` python
    import pytreeclass as pytc
    import dataclasses as dc

    class A(pytc.TreeClass):
        def ___init__(self, a:int):
            self.a = a

    class B(A):
        ...

    ```

    When instantiating `B(a=...)`, an error will be raised, because using `TreeClass` is equivalent of decorating all classes with `@dataclass`, which synthesize the `__init__` method based on the fields.
    Since no fields (e.g. type hinted values) then the synthesized `__init__` method .

    The previous code is equivalent to this code.

    ```python
    @dc.dataclass
    class A:
        def __init__(self, a:int):
            self.a = a
    @dc.dataclass
    class B:
        ...
    ```

2) `dataclass_transform` does not play nicely with user created `__init__` see [1](https://github.com/microsoft/pyright/issues/4738), [2](https://github.com/python/typing/discussions/1187)


#### `leafwise_transform` is decoupled from `TreeClass`.

instead decorate the class with `pytreeclass.leafwise`.


## `PyTreeClass` v0.4

### Changes

1) User-provided `re.Pattern` is used to match keys with regex pattern instead of using `RegexKey`

    <details>

    Example:

    ```python
    import pytreeclass as pytc
    import re 

    tree = {"l1":1, "l2":2, "b":3}
    tree = pytc.AtIndexer(tree)
    tree.at[re.compile("l.*")].get()
    # {'b': None, 'l1': 1, 'l2': 2}
    ```
    </details>

### Deprecations
1) `RegexKey`  is deprecated. use `re` compiled patterns instead.
2) `tree_indent` is deprecated. use `tree_diagram(tree).replace(...)` to replace the edges characters with spaces.

### New features

1)  Add  `tree_mask`, `tree_unmask` to freeze/unfreeze tree leaves based on a callable/boolean pytree mask. defaults to masking non-inexact types by frozen wrapper.
    <details>

    Example: Pass non-`jax` types through `jax` transformation without error.

    ```python
    # pass non-differentiable values to `jax.grad`
    import pytreeclass as pytc
    import jax
    @jax.grad
    def square(tree):
        tree = pytc.tree_unmask(tree)
        return tree[0]**2
    tree = (1., 2)  # contains a non-differentiable node
    square(pytc.tree_mask(tree))
    # (Array(2., dtype=float32, weak_type=True), #2)
    ```

    </details>



2) Support extending match keys by adding abstract base class `BaseKey`. check      docstring for example


3) Support multi-index by any acceptable form. e.g. boolean pytree, key, int, or `BaseKey` instance

    <details>


    Example:

    ```python

    import pytreeclass as pytc
    tree = {"l1":1, "l2":2, "b":3}
    tree = pytc.AtIndexer(tree)
    tree.at["l1","l2"].get()
    # {'b': None, 'l1': 1, 'l2': 2}

    ```
    </details>


4) add `scan` to `AtIndexer` to carry a state while applying a function.
    
    <details>

    Example:

    ```python

    import pytreeclass as pytc
    def scan_func(leaf, state):
        # increase the state by 1 for each function call
        return leaf**2, state+1

    tree = {"l1": 1, "l2": 2, "b": 3}
    tree = pytc.AtIndexer(tree)
    tree, state = tree.at["l1", "l2"].scan(scan_func, 0)
    state
    # 2
    tree
    # {'b': 3, 'l1': 1, 'l2': 4}

    ```
    </details>


5) `tree_summary` improvements.

   - Add size column to `tree_summary`.
   - add `def_count` to dispatch count rule for type.
   - add `def_size` to dispatch size rule for type.
   - add `def_type` to dispatch type display.

    <details>

    Example:

    ```python

    import pytreeclass as pytc
    import jax.numpy as jnp

    x = jnp.ones((5, 5))

    print(pytc.tree_summary([1, 2, 3, x]))
    # ┌────┬────────┬─────┬───────┐
    # │Name│Type    │Count│Size   │
    # ├────┼────────┼─────┼───────┤
    # │[0] │int     │1    │       │
    # ├────┼────────┼─────┼───────┤
    # │[1] │int     │1    │       │
    # ├────┼────────┼─────┼───────┤
    # │[2] │int     │1    │       │
    # ├────┼────────┼─────┼───────┤
    # │[3] │f32[5,5]│25   │100.00B│
    # ├────┼────────┼─────┼───────┤
    # │Σ   │list    │28   │100.00B│
    # └────┴────────┴─────┴───────┘

    # make list display its number of elements
    # in the type row
    @pytc.tree_summary.def_type(list)
    def _(_: list) -> str:
        return f"List[{len(_)}]"

    print(pytc.tree_summary([1, 2, 3, x]))
    # ┌────┬────────┬─────┬───────┐
    # │Name│Type    │Count│Size   │
    # ├────┼────────┼─────┼───────┤
    # │[0] │int     │1    │       │
    # ├────┼────────┼─────┼───────┤
    # │[1] │int     │1    │       │
    # ├────┼────────┼─────┼───────┤
    # │[2] │int     │1    │       │
    # ├────┼────────┼─────┼───────┤
    # │[3] │f32[5,5]│25   │100.00B│
    # ├────┼────────┼─────┼───────┤
    # │Σ   │List[4] │28   │100.00B│
    # └────┴────────┴─────┴───────┘

    ```

    </details>


6) Export pytrees to dot language using `tree_graph`

    <details>

    ```python
    # define custom style for a node by dispatching on the value
    # the defined function should return a dict of attributes
    # that will be passed to graphviz.
    import pytreeclass as pytc
    tree = [1, 2, dict(a=3)]
    @pytc.tree_graph.def_nodestyle(list)
    def _(_) -> dict[str, str]:
        return dict(shape="circle", style="filled", fillcolor="lightblue")
    dot_graph = graphviz.Source(pytc.tree_graph(tree))
    dot_graph
    ```

    ![image](https://github.com/ASEM000/PyTreeClass/assets/48389287/1d5168f0-2696-4d46-bdec-5338b0619605)

7) Add variable position arguments and variable keyword arguments to `pytc.field` `kind`

    <details>

    ```python
    import pytreeclass as pytc


    class Tree(pytc.TreeClass):
        a: int = pytc.field(kind="VAR_POS")
        b: int = pytc.field(kind="POS_ONLY")
        c: int = pytc.field(kind="VAR_KW")
        d: int
        e: int = pytc.field(kind="KW_ONLY")


    Tree.__init__
    # <function __main__.Tree.__init__(self, b: int, /, d: int, *a: int, e: int, **c: int) -> None>
    ```
    </details>


This release introduces lots of `functools.singledispatch` usage, to enable the greater customization.
- `{freeze,unfreeze,is_nondiff}.def_type` to define how to `freeze` a type, how to unfreeze it and whether it is considred nondiff or not. these rules are used by these functions and `tree_mask`/`tree_unmask`.
- `tree_graph.def_nodestyle`, `tree_summary.def_{count,type,size}` for pretty printing customization
- `BaseKey.def_alias` to define type alias usage inside `AtIndexer`/`.at`
- Internally, most of the pretty printing is using dispatching to define repr/str rules for each instance type.