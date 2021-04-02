# Apache 2.0 Jackson Loper 2021
# Modified from https://github.com/tensorflow/custom-op

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

time_two_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_time_two_ops.so'))
basin_finder = time_two_ops.basin_finder
segment_sum_middle_axis = time_two_ops.segment_sum_middle_axis
