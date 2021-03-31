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
import tensorflow as tf

def edt1d(f,axis):
    shp=tf.shape(f)
    start_batch=tf.reduce_prod(tf.shape(f)[:axis])
    nn = shp[axis]
    end_batch = tf.reduce_prod(shp[axis+1:])
    f_reshaped = tf.reshape(f,(start_batch,nn,end_batch))

    out,z,v,basins = basin_finder(f_reshaped)

    return tf.reshape(out,f.shape)