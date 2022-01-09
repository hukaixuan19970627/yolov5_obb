// Modified from
// https://github.com/facebookresearch/detectron2/tree/master/detectron2/layers/csrc/nms_rotated
// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#include <ATen/ATen.h>
#include <torch/extension.h>


#ifdef WITH_CUDA
at::Tensor nms_rotated_cuda(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const float iou_threshold);

at::Tensor poly_nms_cuda(
    const at::Tensor boxes,
    float nms_overlap_thresh);
#endif

at::Tensor nms_rotated_cpu(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const float iou_threshold);


inline at::Tensor nms_rotated(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const float iou_threshold) {
  assert(dets.device().is_cuda() == scores.device().is_cuda());
  if (dets.device().is_cuda()) {
#ifdef WITH_CUDA
    return nms_rotated_cuda(
        dets.contiguous(), scores.contiguous(), iou_threshold);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  return nms_rotated_cpu(dets.contiguous(), scores.contiguous(), iou_threshold);
}


inline at::Tensor nms_poly(
    const at::Tensor& dets,
    const float iou_threshold) {
  if (dets.device().is_cuda()) {
#ifdef WITH_CUDA
    if (dets.numel() == 0)
      return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
    return poly_nms_cuda(dets, iou_threshold);
#else
    AT_ERROR("POLY_NMS is not compiled with GPU support");
#endif
  }
  AT_ERROR("POLY_NMS is not implemented on CPU");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms_rotated", &nms_rotated, "nms for rotated bboxes");
  m.def("nms_poly", &nms_poly, "nms for poly bboxes");
}
