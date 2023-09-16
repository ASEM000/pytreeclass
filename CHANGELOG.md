# Changelog

## v0.10.0

- Supports multibackend:
  - `numpy` + `optree` via `export PYTREECLASS_BACKEND=numpy` (lightweight option)
  - `jax` via `export PYTREECLASS_BACKEND=jax` - The default -

- drop `callback` option in parallel options in `is_parallel`
- Add parallel processing via `is_parallel` to `.{get,set}`


## v0.9.2

## Changes:

-  change `threads_count`  in `apply` parallel kwargs to `max_workers`


## v0.9.1

### Additions:

- Add parallel mapping option in `AtIndexer`. This enables myriad of tasks, like reading a pytree of image file names.

```python
# benchmarking serial vs sequential image read
# on mac m1 cpu with image of size 512x512x3
import pytreeclass as tc
from matplotlib.pyplot import imread
paths = ["lenna.png"] * 10
indexer = tc.AtIndexer(paths)
%timeit indexer[...].apply(imread,is_parallel=True)  # parallel
# 24.9 ms ± 938 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
%timeit indexer[...].apply(imread)  # not parallel
# # 84.8 ms ± 453 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
```


## V0.9

### Breaking changes:
- To simplify the API the following will be removed:
  1) `tree_repr_with_trace`
  2) `tree_map_with_trace`
  3) `tree_flatten_with_trace`
  4) `tree_leaves_with_trace`

    A variant of these will be included in the common recipes.


## V0.8

### Additions:
- Add `on_getattr` in `field` to apply function on `__getattr__`

### Breaking changes:
- Rename `callbacks` in `field` to `on_setattr` to match `attrs` and better reflect its functionality.



_These changes enable:_

1) stricter data validation on instance values, as in the following example:

    <details> 

    `on_setattr` ensure the value is of certain type (e.g.integer) during _initialization_, and `on_getattr`, ensure the value is of certain type (e.g. integer) whenever its accessed.


    ```python

    import pytreeclass as tc
    import jax

    def assert_int(x):
        assert isinstance(x, int), "must be an int"
        return x

    @tc.autoinit
    class Tree(tc.TreeClass):
        a: int = tc.field(on_getattr=[assert_int], on_setattr=[assert_int])

        def __call__(self, x):
            # enusre `a` is an int before using it in computation by calling `assert_int`
            a: int = self.a
            return a + x

    tree = Tree(a=1)
    print(tree(1.0))  # 2.0
    tree = jax.tree_map(lambda x: x + 0.0, tree)  # make `a` a float
    tree(1.0)  # AssertionError: must be an int
    ```

    </details>

2) Frozen field without using `tree_mask`/`tree_unmask`

    <details>

    The following shows a pattern where the value is frozen on `__setattr__` and unfrozen whenever accessed, this ensures that `jax` transformation does not see the value. the following example showcase this functionality

    ```python
    import pytreeclass as tc
    import jax

    @tc.autoinit
    class Tree(tc.TreeClass):
        frozen_a : int = tc.field(on_getattr=[tc.unfreeze], on_setattr=[tc.freeze])

        def __call__(self, x):
            return self.frozen_a + x

    tree = Tree(frozen_a=1)  # 1 is non-jaxtype
    # can be used in jax transformations

    @jax.jit
    def f(tree, x):
        return tree(x)

    f(tree, 1.0)  # 2.0

    grads = jax.grad(f)(tree, 1.0)  # Tree(frozen_a=#1)
    ```

    Compared with other libraies that implements `static_field`, this pattern has *lower* overhead and does not alter `tree_flatten`/`tree_unflatten` methods of the tree.


    </details>

3) Easier way to create a buffer (non-trainable array)

    <details>

    Just use `jax.lax.stop_gradient` in `on_getattr`

    ```python
    import pytreeclass as tc
    import jax
    import jax.numpy as jnp

    def assert_array(x):
        assert isinstance(x, jax.Array)
        return x

    @tc.autoinit
    class Tree(tc.TreeClass):
        buffer: jax.Array = tc.field(on_getattr=[jax.lax.stop_gradient],on_setattr=[assert_array])
        def __call__(self, x):
            return self.buffer**x
        
    tree = Tree(buffer=jnp.array([1.0, 2.0, 3.0]))
    tree(2.0)  # Array([1., 4., 9.], dtype=float32)
    @jax.jit
    def f(tree, x):
        return jnp.sum(tree(x))

    f(tree, 1.0)  # Array([1., 2., 3.], dtype=float32)
    print(jax.grad(f)(tree, 1.0))  # Tree(buffer=[0. 0. 0.])
    ```

    </details>

## v0.7

- Remove `.at` as an alias for `__getitem__` when specifying a path entry for where in `AtIndexer`. This leads to less verbose style.

Example:

```python

>>> tree = {"level1_0": {"level2_0": 100, "level2_1": 200}, "level1_1": 300}
>>> indexer = tc.AtIndexer(tree)

>>> # Before:
>>> # style 1 (with at):
>>> indexer.at["level1_0"].at["level2_0", "level2_1"].get()
{'level1_0': {'level2_0': 100, 'level2_1': 200}, 'level1_1': None}
>>> # style 2 (no at):
>>> indexer["level1_0"]["level2_0", "level2_1"].get()

>>> # After
>>> # only style 2 is valid
>>> indexer["level1_0"]["level2_0", "level2_1"].get()
```

```diff
- tree = indexer.at["level1_0"].at["level2_0", "level2_1"].get()
+ tree = indexer["level1_0"]["level2_0", "level2_1"].get()
```

For `TreeClass`

`at` is specified _once_ for each change

```diff
@tc.autoinit
class Tree(tc.TreeClass):
    a: float = 1.0
    b: tuple[float, float] = (2.0, 3.0)
    c: jax.Array = jnp.array([4.0, 5.0, 6.0])

    def __call__(self, x):
        return self.a + self.b[0] + self.c + x


tree = Tree()
mask = jax.tree_map(lambda x: x > 5, tree)
tree = tree\
       .at["a"].set(100.0)\
-       .at["b"].at[0].set(10.0)\
+      .at["b"][0].set(10.0)\
       .at[mask].set(100.0)
```



## v0.6.0post0
- using `tree_{repr,str}` with an object containing cyclic references will raise `RecursionError` instead of displaying cyclicref.


## v0.6.0
- Allow nested mutations using `.at[method](*args, **kwargs)`.
  After the change, inner methods can mutate **_copied_** new instances at any level not just the top level.
  a motivation for this is to experiment with _lazy initialization scheme_, where inner layers need to mutate their inner state. see the example below for `flax`-like lazy initialization as descriped [here](https://docs.google.com/presentation/d/1ngKWUwsSqAwPRvATG8sAxMzu9ujv4N__cKsUofdNno0/edit#slide=id.g8d686e6bf0_1_57)

  <details>

    ```python

    import pytreeclass as tc
    import jax.random as jr
    from typing import Any
    import jax
    import jax.numpy as jnp
    from typing import Callable, TypeVar

    T = TypeVar("T")

    @tc.autoinit
    class LazyLinear(tc.TreeClass):
        outdim: int
        weight_init: Callable[..., T] = jax.nn.initializers.glorot_normal()
        bias_init: Callable[..., T] = jax.nn.initializers.zeros

        def param(self, name: str, init_func: Callable[..., T], *args) -> T:
            if name not in vars(self):
                setattr(self, name, init_func(*args))
            return vars(self)[name]

        def __call__(self, x: jax.Array, *, key: jr.KeyArray = jr.PRNGKey(0)):
            w = self.param("weight", self.weight_init, key, (x.shape[-1], self.outdim))
            y = x @ w
            if self.bias_init is not None:
                b = self.param("bias", self.bias_init, key, (self.outdim,))
                return y + b
            return y


    @tc.autoinit
    class StackedLinear(tc.TreeClass):
        l1: LazyLinear = LazyLinear(outdim=10)
        l2: LazyLinear = LazyLinear(outdim=1)

        def call(self, x: jax.Array):
            return self.l2(jax.nn.relu(self.l1(x)))

    lazy_layer = StackedLinear()
    print(repr(lazy_layer))
    # StackedLinear(
    #   l1=LazyLinear(
    #     outdim=10, 
    #     weight_init=init(key, shape, dtype), 
    #     bias_init=zeros(key, shape, dtype)
    #   ), 
    #   l2=LazyLinear(
    #     outdim=1, 
    #     weight_init=init(key, shape, dtype), 
    #     bias_init=zeros(key, shape, dtype)
    #   )
    # )

    _, materialized_layer = lazy_layer.at["call"](jnp.ones((1, 5)))
    materialized_layer
    # StackedLinear(
    #   l1=LazyLinear(
    #     outdim=10, 
    #     weight_init=init(key, shape, dtype), 
    #     bias_init=zeros(key, shape, dtype), 
    #     weight=f32[5,10](μ=-0.04, σ=0.32, ∈[-0.74,0.63]), 
    #     bias=f32[10](μ=0.00, σ=0.00, ∈[0.00,0.00])
    #   ), 
    #   l2=LazyLinear(
    #     outdim=1, 
    #     weight_init=init(key, shape, dtype), 
    #     bias_init=zeros(key, shape, dtype), 
    #     weight=f32[10,1](μ=-0.07, σ=0.23, ∈[-0.34,0.34]), 
    #     bias=f32[1](μ=0.00, σ=0.00, ∈[0.00,0.00])
    #   )
    # )
    
    materialized_layer(jnp.ones((1, 5)))
    # Array([[0.16712935]], dtype=float32)
    ```
    </details>


## v0.5post0

- fix `__init_subclass__`. not accepting arguments. this bug is introduced since `v0.5`


## v0.5

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
import pytreeclass as tc
import dataclasses as dc

class Tree(tc.TreeClass):
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
import pytreeclass as tc

@tc.autoinit
class Tree(tc.TreeClass):
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
    import pytreeclass as tc
    import dataclasses as dc

    class A(tc.TreeClass):
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


## `pytreeclass` v0.4

### Changes

1) User-provided `re.Pattern` is used to match keys with regex pattern instead of using `RegexKey`

    <details>

    Example:

    ```python
    import pytreeclass as tc
    import re 

    tree = {"l1":1, "l2":2, "b":3}
    tree = tc.AtIndexer(tree)
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
    import pytreeclass as tc
    import jax
    @jax.grad
    def square(tree):
        tree = tc.tree_unmask(tree)
        return tree[0]**2
    tree = (1., 2)  # contains a non-differentiable node
    square(tc.tree_mask(tree))
    # (Array(2., dtype=float32, weak_type=True), #2)
    ```

    </details>



2) Support extending match keys by adding abstract base class `BaseKey`. check      docstring for example


3) Support multi-index by any acceptable form. e.g. boolean pytree, key, int, or `BaseKey` instance

    <details>


    Example:

    ```python

    import pytreeclass as tc
    tree = {"l1":1, "l2":2, "b":3}
    tree = tc.AtIndexer(tree)
    tree.at["l1","l2"].get()
    # {'b': None, 'l1': 1, 'l2': 2}

    ```
    </details>


4) add `scan` to `AtIndexer` to carry a state while applying a function.
    
    <details>

    Example:

    ```python

    import pytreeclass as tc
    def scan_func(leaf, state):
        # increase the state by 1 for each function call
        return leaf**2, state+1

    tree = {"l1": 1, "l2": 2, "b": 3}
    tree = tc.AtIndexer(tree)
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

    import pytreeclass as tc
    import jax.numpy as jnp

    x = jnp.ones((5, 5))

    print(tc.tree_summary([1, 2, 3, x]))
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
    @tc.tree_summary.def_type(list)
    def _(_: list) -> str:
        return f"List[{len(_)}]"

    print(tc.tree_summary([1, 2, 3, x]))
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
    import pytreeclass as tc
    tree = [1, 2, dict(a=3)]
    @tc.tree_graph.def_nodestyle(list)
    def _(_) -> dict[str, str]:
        return dict(shape="circle", style="filled", fillcolor="lightblue")
    dot_graph = graphviz.Source(tc.tree_graph(tree))
    dot_graph
    ```

    ![image](https://github.com/ASEM000/pytreeclass/assets/48389287/1d5168f0-2696-4d46-bdec-5338b0619605)

7) Add variable position arguments and variable keyword arguments to `tc.field` `kind`

    <details>

    ```python
    import pytreeclass as tc


    class Tree(tc.TreeClass):
        a: int = tc.field(kind="VAR_POS")
        b: int = tc.field(kind="POS_ONLY")
        c: int = tc.field(kind="VAR_KW")
        d: int
        e: int = tc.field(kind="KW_ONLY")


    Tree.__init__
    # <function __main__.Tree.__init__(self, b: int, /, d: int, *a: int, e: int, **c: int) -> None>
    ```
    </details>


This release introduces lots of `functools.singledispatch` usage, to enable the greater customization.
- `{freeze,unfreeze,is_nondiff}.def_type` to define how to `freeze` a type, how to unfreeze it and whether it is considred nondiff or not. these rules are used by these functions and `tree_mask`/`tree_unmask`.
- `tree_graph.def_nodestyle`, `tree_summary.def_{count,type,size}` for pretty printing customization
- `BaseKey.def_alias` to define type alias usage inside `AtIndexer`/`.at`
- Internally, most of the pretty printing is using dispatching to define repr/str rules for each instance type.