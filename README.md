<h1 align="center" >🌲pytreeclass🌲</h1>
<h2 align="center">Write pytorch-like layers with keras-like visualizations in JAX.</h2>

[**Installation**](#Installation)
|[**Description**](#Description)
|[**Examples**](#Examples)

![Tests](https://github.com/ASEM000/pytreeclass/actions/workflows/tests.yml/badge.svg)
![pyver](https://img.shields.io/badge/python-3.7%203.8%203.9%203.10-red)
![codestyle](https://img.shields.io/badge/code%20style-yapf-lightgrey)

<!-- [![Downloads](https://static.pepy.tech/personalized-badge/kernex?period=month&units=international_system&left_color=black&right_color=blue&left_text=Downloads)](https://pepy.tech/project/kernex) -->

## 🛠️ Installation<a id="Installation"></a>

```python
pip install
```

## 📖 Description<a id="Description"></a>

A JAX compatible `dataclass` like datastructure with the following functionalities

- Create PyTorch like NN classes like [equinox](https://github.com/patrick-kidger/equinox) and [Treex](https://github.com/cgarciae/treex)
- Provides Keras-like `model.summary()` and `plot_model` visualizations for pytrees wrapped with `treeclass`.
- Apply math/numpy operations like [tree-math](https://github.com/google/tree-math)
- Registering user-defined reduce operations on each class.
- Some fancy indexing syntax functionalities like `x[x>0]` on pytrees

## 🔢 Examples<a id="Examples"></a>

<details><summary>Write PyTorch like NN classes</summary>

```python
# construct a Pytorch like NN classes with JAX

@treeclass
class Linear :

 weight : jnp.ndarray
 bias   : jnp.ndarray

 def __init__(self,key,in_dim,out_dim):
   self.weight = jax.random.normal(key,shape=(in_dim, out_dim)) * jnp.sqrt(2/in_dim)
   self.bias = jnp.ones((1,out_dim))

 def __call__(self,x):
   return x @ self.weight + self.bias

@treeclass
class StackedLinear:
   l1 : Linear
   l2 : Linear
   l3 : Linear

   def __init__(self,key,in_dim,out_dim):

       keys= jax.random.split(key,3)

       self.l1 = Linear(key=keys[0],in_dim=in_dim,out_dim=128)
       self.l2 = Linear(key=keys[1],in_dim=128,out_dim=128)
       self.l3 = Linear(key=keys[2],in_dim=128,out_dim=out_dim)

   def __call__(self,x):
       x = self.l1(x)
       x = jax.nn.tanh(x)
       x = self.l2(x)
       x = jax.nn.tanh(x)
       x = self.l3(x)

       return x


x = jnp.linspace(0,1,100)[:,None]
y = x**3 + jax.random.uniform(jax.random.PRNGKey(0),(100,1))*0.01

model = StackedLinear(in_dim=1,out_dim=1,key=jax.random.PRNGKey(0))

def loss_func(model,x,y):
   return jnp.mean((model(x)-y)**2 )

@jax.jit
def update(model,x,y):
   value,grads = jax.value_and_grad(loss_func)(model,x,y)
   # no need to use `jax.tree_map` to update the model
   #  as it model is wrapped by treeclass
   return value , model-1e-3*grads

for _ in range(1,2001):
   value,model = update(model,x,y)

plt.scatter(x,model(x),color='r',label = 'Prediction')
plt.scatter(x,y,color='k',label='True')
plt.legend()

```

![image](assets/regression_example.png)

</details>

<details> <summary>Visualize</summary>

```python
>>> print(kernex.viz.summary(model))
┌──────┬───────┬─────────┬───────────────────┐
│Type  │Param #│Size     │Config             │
├──────┼───────┼─────────┼───────────────────┤
│Linear│256    │1.000 KB │bias=f32[1,128]    │
│      │       │         │weight=f32[1,128]  │
├──────┼───────┼─────────┼───────────────────┤
│Linear│16,512 │64.500 KB│bias=f32[1,128]    │
│      │       │         │weight=f32[128,128]│
├──────┼───────┼─────────┼───────────────────┤
│Linear│129    │516.000 B│bias=f32[1,1]      │
│      │       │         │weight=f32[128,1]  │
└──────┴───────┴─────────┴───────────────────┘
Total params :	16,897
Inexact params:	16,897
Other params:	0
----------------------------------------------
Total size :	66.004 KB
Inexact size:	66.004 KB
Other size:	0.000 B
==============================================

>>> print(kernex.viz.tree_box(model,array=x))
# using jax.eval_shape (no-flops operation)
┌──────────────────────────────────────┐
│StackedLinear(Parent)                 │
├──────────────────────────────────────┤
│┌────────────┬────────┬──────────────┐│
││            │ Input  │ f32[100,1]   ││
││ Linear(l1) │────────┼──────────────┤│
││            │ Output │ f32[100,128] ││
│└────────────┴────────┴──────────────┘│
│┌────────────┬────────┬──────────────┐│
││            │ Input  │ f32[100,128] ││
││ Linear(l2) │────────┼──────────────┤│
││            │ Output │ f32[100,128] ││
│└────────────┴────────┴──────────────┘│
│┌────────────┬────────┬──────────────┐│
││            │ Input  │ f32[100,128] ││
││ Linear(l3) │────────┼──────────────┤│
││            │ Output │ f32[100,1]   ││
│└────────────┴────────┴──────────────┘│
└──────────────────────────────────────┘

>>> print(kernex.viz.tree_diagram(model))

StackedLinear
    ├── l1=Linear
    │   ├── weight=f32[1,128]
    │   └── bias=f32[1,128]
    ├── l2=Linear
    │   ├── weight=f32[128,128]
    │   └── bias=f32[1,128]
    └──l3=Linear
        ├── weight=f32[128,1]
        └── bias=f32[1,1]

```

</details>

<details>
<summary>Perform Math operations on JAX pytrees</summary>

```python
from kernex import treeclass,static_field
import jax
from jax import numpy as jnp

@treeclass
class Test :
  a : float
  b : float
  c : float
  name : str = static_field() # ignore from jax computations


# basic operations
A = Test(10,20,30,'A')
assert (A + A) == Test(20,40,60,'A')
assert (A - A) == Test(0,0,0,'A')
assert (A*A).reduce_mean() == 1400
assert (A + 1) == Test(11,21,31,'A')

# selective operations

# only add 1 to field `a`
# all other fields are set to None and returns the same class
assert (A['a'] + 1) == Test(11,None,None,'A')

# use `|` to merge classes by performing ( left_node or  right_node )
Aa = A['a'] + 10 # Test(a=20,b=None,c=None,name=A)
Ab = A['b'] + 10 # Test(a=None,b=30,c=None,name=A)

assert (Aa | Ab | A ) == Test(20,30,30,'A')

# indexing by class
assert A[A>10]  == Test(a=None,b=20,c=30,name='A')


# Register custom operations
B = Test([10,10],20,30,'B')
B.register_op( func=lambda node:node+1,name='plus_one')
assert B.plus_one() == Test(a=[11, 11],b=21,c=31,name='B')


# Register custom reduce operations ( similar to functools.reduce)
C = Test(jnp.array([10,10]),20,30,'C')

C.register_op(
    func=jnp.prod,            # function applied on each node
    name='product',           # name of the function
    reduce_op=lambda x,y:x*y, # function applied between nodes (accumulated * current node)
    init_val=1                # initializer for the reduce function
                )

# product applies only on each node
# and returns an instance of the same class
assert C.product() == Test(a=100,b=20,c=30,name='C')

# `reduce_` + name of the registered function (`product`)
# reduces the class and returns a value
assert C.reduce_product() == 60000
```

</details>
