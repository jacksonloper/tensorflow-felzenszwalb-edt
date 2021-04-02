# What this package is

This package provides a single differentiable tensorflow function
for computing euclidean distance transforms: tensorflow_felzenszwalb_edt.edt1d.

```
edt1d
Input:
- f, a float32 tensor of shape M0 x M1 x M2 ... Mn
- axis, an integer in {0,1,2,...n}

Output is
- g -- a float32/float64 of the same shape as f
- basins -- an int32 of the same shape as f,

satisfying

g[i_0,i_1,...i_{axis-1},p,i_{axis+1}...i_n]
    =
min_q ((q-p)**2 + f[i_0,i_1,...i_{axis-1},q,i_{axis+1}...i_n])

basins[i_0,i_1,...i_{axis-1},p,i_{axis+1}...i_n]
    =
argmin_q ((q-p)**2 + f[i_0,i_1,...i_{axis-1},q,i_{axis+1}...i_n])

```

It also provides one additional convenience function:

```
gather_with_batching

Input
- params, a tensor
- indices, an integer tensor of the same shape as params
- axis, an integer

The output satisfies

output[p0,p1...p_{axis-1},i,p_{axis+1}...p_{N-1}]

        =

params[p0,p1,...p_{axis-1},
            indices[p0,p1,...p_{axis-1},i,p_{axis+1}...p_{N-1}],
        p_{axis+1},....p_{N-1}
      ]

For example, with N=3 and axis=1, we have

output[i0,i1,i2] = params[i0,indices[i0,i1,i2],i2]
```

This function is a small wrapper around `tf.gather` that is useful for computing the argmin over the edt operator in the multidimensional case (see "getting the basin locations", below).
```

## example usage: multiple dimensions

Recall that the euclidean distance transform is separable, so if you want to compute
a multidimensional distance transform, you can perform it iteratively.  For example
if you do this:

```
import numpy.random as npr
f=npr.randn(10,20,40)
g=edt1d(f,axis=0)[0]
g=edt1d(g,axis=1)[0]
g=edt1d(g,axis=2)[0]
```

then g will satisfy

```
g[i0,i1,i2] = min_{j0,j1,j2} ((i0-j0)**2 + (i1-j1)**2 + (i2-j2)**2 + f[j0,j1,j2])
```

## example usage: morphological edt

One special case of edt is the so-called morphological edt.  Given
a binary vector `b`, the task is to compute objects such as

```
g[i0,i1,i2] = min_{j0,j1,j2: b[j0,j1,j2]=True} ((i0-j0)**2 + (i1-j1)**2 + (i2-j2)**2)
```

This can be achieved by taking constructing a float32 array where the false values are
sufficiently large:

```
import numpy.random as npr
b = npr.randn(10,20,40)>3
f = (~b)*np.sum(np.array(b.shape)**2)
d = edt1d(f,0)[0]
d = edt1d(d,1)[0]
d = edt1d(d,2)[0]
```

## example usage: batch processing

Say you have many 3d images of the same size and you would like to perform the edt over each of them.  To do so, edt1d across the image axes, and leave the batch axis alone:

```
import numpy.random as npr
b = npr.randn(1000,10,20,40) # 1000 images, each of size 10x20x40
d = edt1d(f,1)[0]
d = edt1d(d,2)[0]
d = edt1d(d,3)[0]
```

## example usage: getting the basin locations

Above we have considered methods to compute g such that

```
g_{i0,i1,i2} = min_{a,b,c} ((i0-a)^2 + (i1-b)^2+(i2-c)^2 + f(a,b,c))
```

But in some cases one may want to find the optimizing parameters for this minimization problem, i.e.

```
argmin_{a,b,c} ((i0-a)^2 + (i1-b)^2+(i2-c)^2 + f(a,b,c))
```

Put another way, we may wish to obtaining the parameters a,b,c which yield the values of g.  To obtain these parameters for each value of i0,i1,i2, we can use the "basins" output of edt1d and some gather operations.  For example:

```
g0,b0=tensorflow_felzenszwalb_edt.edt1d(f,axis=0)

g1,b1=tensorflow_felzenszwalb_edt.edt1d(g0,axis=1)
i0starstar = tensorflow_felzenszwalb_edt.gather_with_batching(b0,b1,axis=1)

g2,b2=tensorflow_felzenszwalb_edt.edt1d(g1,axis=2)
i2star = b2
i1star = gather_with_batching(b1,b2,axis=2)
i0star = gather_with_batching(i0starstar,b2,axis=2)
```

After running this code, for each i0,i1,i2, the values of `i0star[i0,i1,i2],i1star[i0,i1,i2],i2star[i0,i1,i2]` hold the values of (a,b,c) which solve the edt problem at that location.  This same idea can be generalized to 4d edt or higher.

The 2d case is even simpler:

```
g0,b0=tensorflow_felzenszwalb_edt.edt1d(f,axis=0)

g1,b1=tensorflow_felzenszwalb_edt.edt1d(g0,axis=1)
i1star = b1
i0star = tensorflow_felzenszwalb_edt.gather_with_batching(b0,b1,axis=1)
```


# Installation instructions

```
pip install --upgrade https://github.com/jacksonloper/tensorflow-felzenszwalb-edt/raw/master/artifacts/tensorflow_felzenszwalb_edt-0.0.1-py37-none-linux_x86_64.whl
```

# Build instructions

If you would like to make improvements to this package, you will
need to be able to build it.  Here is the magic sauce for building in the docker environment:

```
cd [[path to tensorflow-felzenszwalb-edt]]
docker run --rm -it --mount type=bind,source=`pwd`,target=/app tensorflow/tensorflow:custom-op-gpu-ubuntu16 /bin/bash
cd app
./configure [[ answer YES and YES! ]]
make time_two_op
make pip_pkg
```

This will build the necessary python extensions in the tensorflow_felzenszwalb_edt
directory and also build a wheel which will be deposited into the artifacts directory.

The package can be tested directly by importing this package from a python
script run in the base directory of this project.  E.g. load the test.ipynb notebook and try it out.

```
cd [[path to tensorflow-felzenszwalb-edt]]
python tensorflow_felzenszwalb_edt/python/ops/time_two_ops_test.py
```

If your modifications to the package are working well, the wheel can be installed by
```
pip install artifacts/tensorflow_felzenszwalb_edt-0.0.1-py37-none-linux_x86_64.whl
```
