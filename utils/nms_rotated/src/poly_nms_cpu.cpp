#include <torch/extension.h>

template <typename scalar_t>
at::Tensor poly_nms_cpu_kernel(const at::Tensor& dets, const float threshold) {

