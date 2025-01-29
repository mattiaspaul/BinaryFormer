#ifndef HAMMINGATTENTION_H
#define HAMMINGATTENTION_H
#include <ATen/core/Tensor.h>
#include <torch/torch.h>
#include <torch/script.h>

#include <ATen/core/Tensor.h>
#include <ATen/div_rtn.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/DispatchStub.h>
#include <c10/util/irange.h>

namespace at::native {
namespace mps {
void tiledHammingAttentionReLU(const Tensor&,const Tensor&,const Tensor&,const Tensor& );
void tiledSDPAfloatReLU(const Tensor&,const Tensor&,const Tensor&,const Tensor& );
void tiledSDPAfloat(const Tensor&,const Tensor&,const Tensor&,const Tensor& );
void packbitTensor(const Tensor&, const Tensor& );
void packbit8Tensor(const Tensor&, const Tensor& );

void multiplyTensor(const Tensor&,const Tensor&,const Tensor& );
void hammingTensor(const Tensor&,const Tensor&,const Tensor& );
void matmulTensor(const Tensor&, const Tensor&, const Tensor&);
}
}
#endif
