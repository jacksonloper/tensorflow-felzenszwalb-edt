/* Apache 2.0 Jackson Loper 2021
Modified from https://github.com/tensorflow/custom-op*/
#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "time_two.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"


#include <cmath>
const float verybig = INFINITY;

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;


// Define the CUDA kernel.
template <typename T, typename S>
__device__ float calcint(S q1,S q2,T f1,T f2){
    T q1f = static_cast<T>(q1);
    T q2f = static_cast<T>(q2);
    return ((f1+q1f*q1f) - (f2+q2f*q2f)) / (2*q1f - 2*q2f);
}


template <typename T, typename S>
__global__ void BasinFinderCudaKernel(const int dim0, const int dim1,
    const int dim2, const T* f, T* out, T* z, S* v, S* basins) {

        const int batchdim = threadIdx.x + blockDim.x*blockIdx.x;
        const int i0 = batchdim/dim2;
        const int i2 = batchdim%dim2;
        // f is a 3-tensor of shape (dim0,dim1,dim2)
        // this thread looks at f[batchdim//shape[2],:,batchdim%shape[2]]

        if((i0<dim0)&&(i2<dim2)) {

          const int offset1= i0*dim1*dim2+i2;
          const int offset2= i0*(dim1+1)*dim2+i2;



          // initialize v,z
          for(int i1=0; i1<dim1; i1++) {
            v[offset1+i1*dim2]=0;
            z[offset2+i1*dim2]=0;
          }
          z[offset2+dim1*dim2]=0;

          // compute lower parabolas
          int k=0;
          z[offset2+0]=-verybig;
          z[offset2+dim2]=verybig;

          for(int q=1; q<dim1; q++) {
                //printf("%d %d %d :: %d %d \n",i0,i2,q,offset1+k*dim2,offset1+v[offset1+k*dim2]*dim2);
                float s=calcint<T,S>(q,v[offset1+k*dim2],f[offset1+q*dim2],f[offset1+v[offset1+k*dim2]*dim2]);
                //printf("%d %d %d :: %d %d %f \n",i0,i2,q,offset1+k*dim2,offset1+v[offset1+k*dim2]*dim2,s);

                while(s<=z[offset2+k*dim2]){
                    k=k-1;
                    //printf("%d %d %d :: %d %d \n",i0,i2,q,offset1+k*dim2,offset1+v[offset1+k*dim2]*dim2);
                    s=calcint<T,S>(q,v[offset1+k*dim2],f[offset1+q*dim2],f[offset1+v[offset1+k*dim2]*dim2]);
                    //printf("%d %d %d :: %d %d %f \n",i0,i2,q,offset1+k*dim2,offset1+v[offset1+k*dim2]*dim2,s);
                }
                k=k+1;
                v[offset1+k*dim2]=q;
                z[offset2+k*dim2]=s;
                z[offset2+k*dim2+dim2]=verybig;
          }

          // compute basins and out
          k=0;
          for(int q=0; q<dim1; q++) {
            while(z[offset2+(k+1)*dim2]<q) {
              k=k+1;
            }
            int thisv=v[offset1+k*dim2];
            basins[offset1+q*dim2]=thisv;
            out[offset1+q*dim2] = (q-thisv)*(q-thisv) + f[offset1+thisv*dim2];
          }
        }
}


template <typename T, typename S>
__global__ void SegmentSumMiddleAxisCudaKernel(const int dim0, const int dim1,
    const int dim2, const T* weights, const S* basins, T* out) {

        const int batchdim = threadIdx.x + blockDim.x*blockIdx.x;
        const int i0 = batchdim/dim2;
        const int i2 = batchdim%dim2;
        // f is a 3-tensor of shape (dim0,dim1,dim2)
        // this thread looks at f[batchdim//shape[2],:,batchdim%shape[2]]

        if((i0<dim0)&&(i2<dim2)) {

          const int offset1= i0*dim1*dim2+i2;
          const int offset2= i0*(dim1+1)*dim2+i2;

          for(int i1=0; i1<dim1; i1++) {
            out[offset1+i1*dim2]=0;
          }

          // add up the weights
          for(int p=0; p<dim1; p++) {
            int myq = basins[offset1+p*dim2];
            out[offset1+myq*dim2]+=weights[offset1+p*dim2];
          }
        }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T,typename S>
struct BasinFinderFunctor<GPUDevice, T,S> {
  void operator()(const GPUDevice& d, int dim0, int dim1, int dim2, const T* f, T* out, T* z, S* v, S* basins) {
    // Launch the cuda kernel.
    //
    // See core/util/cuda_kernel_helper.h for example of computing
    // block count and thread_per_block count.
    int block_count = 1+static_cast<int>(dim0*dim2/8);
    int thread_per_block = 8;
    BasinFinderCudaKernel<T,S>
        <<<block_count, thread_per_block, 0, d.stream()>>>(dim0, dim1,dim2, f, out,z,v,basins);
  }
};


// Define the GPU implementation that launches the CUDA kernel.
template <typename T,typename S>
struct SegmentSumMiddleAxisFunctor<GPUDevice, T,S> {
  void operator()(const GPUDevice& d, int dim0, int dim1, int dim2, const T* in, const S* basins, T* out) {
    // Launch the cuda kernel.
    //
    // See core/util/cuda_kernel_helper.h for example of computing
    // block count and thread_per_block count.
    int block_count = 1+static_cast<int>(dim0*dim2/8);
    int thread_per_block = 8;

    SegmentSumMiddleAxisCudaKernel<T,S>
        <<<block_count, thread_per_block, 0, d.stream()>>>(dim0, dim1,dim2, in, basins,out);
  }
};

// Explicitly instantiate functors for the types of OpKernels registered.
template struct BasinFinderFunctor<GPUDevice, float,int32>;
template struct BasinFinderFunctor<GPUDevice, double,int32>;
template struct SegmentSumMiddleAxisFunctor<GPUDevice, float,int32>;
template struct SegmentSumMiddleAxisFunctor<GPUDevice, double,int32>;
}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
