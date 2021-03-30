/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("BasinFinder")
    .Attr("T: {float}")
    .Input("in: T")
    .Output("out: T")      // actual edt
    .Output("z: T")        // different lower-bound parabola intersections
    .Output("v: int32")        // indices of lower-bound parabola centers
    .Output("basins: int32")   // basins
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &input)); // insist input is rank 3

      // first output same as the inmput
      c->set_output(0, c->input(0));

      // z has axis1 to be 1 longer
      ::tensorflow::shape_inference::DimensionHandle axis1_dim = c->Dim(c->input(0),1);
      const int64 axis1_dim_int = c->Value(axis1_dim);
      ::tensorflow::shape_inference::DimensionHandle axis1_newdim;
      if(axis1_dim_int==::tensorflow::shape_inference::InferenceContext::kUnknownDim) {
        axis1_newdim = c->UnknownDim();
      } else {
        axis1_newdim = c->MakeDim(axis1_dim_int+1);
      }
      c->set_output(1, c->MakeShape({c->Dim(c->input(0),0),axis1_newdim,c->Dim(c->input(0),2)}));

      // v,basins same as the input
      c->set_output(2, c->input(0));
      c->set_output(3, c->input(0));

      // done!
      return Status::OK();
    });
