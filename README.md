# What this package is

This package provides a single differentiable tensorflow function,
tensorflow_felzenszwalb_edt.edt1d.

```
edt1d
Input:
- f, a float32 tensor of shape M0 x M1 x M2 ... Mn
- axis, an integer in {0,1,2,...n}

Output is a float32 tensor g of the same shape as f, satisfying

  g[i_0,i_1,...i_{axis-1},p,i_{axis+1}...i_n]
    =
  min_q ((q-p)**2 + f[i_0,i_1,...i_{axis-1},q,i_{axis+1}...i_n])

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
