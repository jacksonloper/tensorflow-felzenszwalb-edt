# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""TensorFlow GPU custom op example."""

from __future__ import absolute_import

from tensorflow_felzenszwalb_edt.python.ops.time_two_ops import basin_finder
from tensorflow_felzenszwalb_edt.python.ops.time_two_ops import segment_sum_middle_axis
import tensorflow as tf

@tf.custom_gradient
def edt1d(f,axis):
    '''
    Input:
    - f, a float32/float64 tensor of shape M0 x M1 x M2 ... Mn
    - axis, an integer in {0,1,2,...n}

    Output is
    - g -- a float32/float64 of the same shape as f
    - b -- an int32 of the same shape as f,

    satisfying

    g[i_0,i_1,...i_{axis-1},p,i_{axis+1}...i_n]
        =
    min_q ((q-p)**2 + f[i_0,i_1,...i_{axis-1},q,i_{axis+1}...i_n])

    b[i_0,i_1,...i_{axis-1},p,i_{axis+1}...i_n]
        =
    argmin_q ((q-p)**2 + f[i_0,i_1,...i_{axis-1},q,i_{axis+1}...i_n])

    '''

    shp=tf.shape(f)
    start_batch=tf.reduce_prod(tf.shape(f)[:axis])
    nn = shp[axis]
    end_batch = tf.reduce_prod(shp[axis+1:])
    f_reshaped = tf.reshape(f,(start_batch,nn,end_batch))

    out,z,v,basins = basin_finder(f_reshaped)

    # basins[i0,p,i2] gives the argmin of the edt problem, i.e.
    # basins[i0,p,i2] = argmin_q (q-p)**2 + f[i0,q,i2])
    # out[i0,p,i2] = (basins[i0,p,i2]-p)**2 + f[i0,basins[i0,p,i2],i2])


    def grad(weights,weightsb):
        '''
        We need tf.math.segment_sum, but batched over indices i0 and i2.
        This only is possible without ragged arrays becaused we
        guarantee that the max value of basins is less than weights.shape[1]
        for each batch.
        '''

        if weightsg is None:
            return None

        weights_reshaped = tf.reshape(weights,(start_batch,nn,end_batch))
        jv=segment_sum_middle_axis(weights_reshaped,basins)

        return tf.reshape(jv,weights.shape)


    return (tf.reshape(out,f.shape),(tf.reshape(basins,f.shape))),grad

def gather_with_batching(params,indices,axis):
    '''
    output[p0,p1...p_{axis-1},i,p_{axis+1}...p_{N-1}]

            =

    params[p0,p1,...p_{axis-1},
               indices[p0,p1,...p_{axis-1},i,p_{axis+1}...p_{N-1}],
           p_{axis+1},....p_{N-1}
          ]

    For example, with N=3 and axis=1, we have

    output[i0,i1,i2] = params[i0,indices[i0,i1,i2],i2]
    '''

    import numpy as np
    n=len(params.shape)
    perm=np.r_[0:n]
    perm[axis],perm[n-1]=perm[n-1],perm[axis]

    params=tf.transpose(params,perm)
    indices=tf.transpose(indices,perm)

    result=tf.gather(params,indices,axis=n-1,batch_dims=n-1)

    return tf.transpose(result,perm)