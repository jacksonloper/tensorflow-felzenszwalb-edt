// kernel_example.h
#ifndef KERNEL_BASIN_FINDER_H_
#define KERNEL_BASIN_FINDER_H_

namespace tensorflow {

namespace functor {

template <typename Device, typename T, typename S>
struct BasinFinderFunctor {
  void operator()(const Device& d, int dim0, int dim1, int dim2, const T* f, T* out, T* z, S* v, S* basins);
};

}  // namespace functor

}  // namespace tensorflow

#endif //KERNEL_BASIN_FINDER_H_
