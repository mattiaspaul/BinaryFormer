# BinaryFormer
MIDL2025 Submission on using Hamming distances in Self-Attention for Long Range Vision Transformers

## Overview
⚡️BinaryFormer aims to reduce computational and memory demand of long-range transformers by reducing the precision of the large matrix multiplication of queries and keys from floating point to binary. 
🚀 It is also the first method to propose a binary backward computation making BinaryFormer models not only efficient at inference but also during training. Key to make this possible is a decomposition of QK^T into a non-differentiable Hamming distance and a (scalar) differentiable weighing based on the binarisation difference. 
🎉 This repository provides the following features:
- [x] Concept of binary in pytorch self-attention with custom backward path
- [x] Efficient CPU implementation using numpy-2.2's ``bitwise_count``  
- [x] Model variants: base, sign (adhoc STE), sign_sg (stop grad.), Hamming, HammingG (with group-linear), HammingB (w/o learned weights)
- [x] Experimental settings for pixel-transformer CIFAR, fine-tuning DINO ViT-S8 segmentation and 3D VQ-diffusion models 
- [x] MPSGraph (Apple GPU) ``HammingDistanceWithPrimaryTensor`` implementation of Hamming-Attention including custom bitpacking
- [ ] Todo: Triton/Cutlass kernels ``wmma_binary_gemm.cu`` that leverage TensorCores and deeper integration into pytorch

## Motivation
While weight quantisation for improved storage and inference of deep transformer models is commonplace, much resources are wasted to compute QK^T with floating point precision (Nvidias Transformer Engine / Hopper is specifically build to enable FP8 in this step). By enabling a binary computation in both forward and backward step theoretical speed-ups and efficiency gains of 16x are possible with little to no sacrifice in performance. This is particularly important for long-range transformers (e.g. in 3D medical imaging) with token lenghts of a thousand and more. The following chart demonstrates the reduction of multiply-accumulate operations (MACs) with batch-size=1, N=2’048, D=384 and 6 heads. Combining a 4x reduction of the value tensor with the proposed Hamming Attention leads to a substantial complexity reduction, where now the MLP (with 4x channel expansion) dominates.
![midl2025_concept_mac](https://github.com/user-attachments/assets/302cb3f6-ad3a-42f1-b8b3-15d7804dfb95)

## Method
We consider the following setting where the input is linearly mapped to keys and queries (with trainable weight matrices) followed by a (softer) hyperbolic tangent activation to yield $\hat{q}$ and $\hat{k}$. (The values are mapped to 4x less channels to further reduce complexity.) Next the sign function is applied to obtain binary-valued (only -1 or +1) $Q^{(b)}$ and $K^{(b)}$. We concatenate $\hat{q}$ and $Q^{(b)}$ before feeding it into $\theta_Q$ and $\theta_K$ small MLPs with groups=#heads that predicts one scalar weight per head/token to produce $q_w$ and $k_w$. Note that we assume ``D=384``, ``H=6`` and the dimensions of the weights will be ``B*H x N x 1`` whereas $Q^{(b)}$ is currently ``B*H x N x 64``.   
```
self.value = nn.Linear(384,384//4) #we reduce the channels in value to further reduce complexity
#learn to predict one scalar weight per head/token
self.theta_Q = nn.Sequential(GroupedLinear(384*2,192),nn.GELU(),GroupedLinear(192,6)) 
self.theta_K = nn.Sequential(GroupedLinear(384*2,192),nn.GELU(),GroupedLinear(192,6))
...
q,k,v = self.query(x),self.key(x),self.value(x) #query,key,value are trainable weights
q_,k_ = self.activate(q),self.activate(k) #activate is a soft version of torch.tanh
Q_b,K_b = torch.sign(q).detach(),torch.sign(k).detach() #no grad necessary
q_w = self.theta_Q(torch.cat((q_,Q_b),-1)) #learned scalar weight/head
k_w = self.theta_K(torch.cat((k_,K_b),-1)) # " = "
```
Now we can compute the multi-head self-attention as follows
```
def forward(ctx, Q_b,q_w,K_b,k_w,V):
    D = 64
    Q_,K_ = packbits(Q_b),packbits(K_b)
    S1 = popcount(Q_^K_.transpose(-2,-1)).to(V).sub(32).mul(-2)/ D**0.5
    S = S1 * q_w * k_w.transpose(-2,-1)
    P = torch.softmax(S,-1)
    O = P @ V
    ctx.save_for_backward(V,S1,P,q_w,k_w)
```
Note that the expensive matrix multiplication ``Q @ K.transpose(-2,-1)`` is no longer required and the output is approximated using the Hamming distance (XNOR + popcount + substration/multiply to match inner-product) and a pointwise multiplication. Packbits converts the ``B*H x N x 64`` bool tensors to Int64 ones with shapes ``B*H x N x 1``. The backward path is slightly more involved:
```
def backward(ctx, dO):
    V,S1,P,q_w,k_w =  ctx.saved_tensors
    dV = P.transpose(-2,-1) @ dO
    dP = dO @ V.transpose(-2,-1)
    D_ = (P*dP).sum(-1,keepdim=True)
    dS =  P*(dP-D_)
    dq_w = (dS*S1*k_w.transpose(-2,-1)).sum(-1,keepdim=True)
    dk_w = (dS*S1*q_w).sum(-2).unsqueeze(-1)
    return None,dq_w,None,dk_w,dV
```
We follow the derivation for the backward path as e.g. described in (https://arxiv.org/abs/2205.14135), but can remove the expensive matrix multiplications ``dQ = dS @ K / D**.5`` and ``dK = dS.T @ Q / D**.5`` in favour of cheap pointwise multiplies. Note that $Q^{(b)}$ and $K^{(b)}$ receive no gradients but only their weights. For a more memory-efficient (IO-aware) implementation the concept of flash attention is also easily applicable by recomputing the individual tiles of $S1$ on-the-fly with Hamming distances during backpropagation. You can find a fully functional replacement of ``F.scaled_dot_product_attention`` in HammingSDPA.py (for CPU with numpy bitcount). We also provide an MPSGraph implementation for efficient training/inference on Apple Silicon (including flash-attention style memory tiling). Custom cutlass/Triton kernels for CUDA devices are work-in-progress.

## Expected efficiency gains
According to the Nvidia documentation and Cutlass profiler results TensorCores should yield a 16x speed-up for binary matrix multiplications (convolutions) compared to FP16. However, the $QK^T$ attention computation is only a part of the whole layer  Unfortunately, neither pytorch nor the Cutlass python package expose 1-bit precision (so far there is only experimental INT4 support) - hence custom kernels are required. numpy-2.2 (for CPU) and MPSGraph (for Apple GPUs) enable efficient 1-bit Hamming distances. 🚀 Depending on sequence length we realise real-world speed ups of up to 8x fold along with substantial (>4x) gains in memory usage. Note, however, that flash-attention implementations are similarly/complementary able to reduce memory impact.  



## Abstract

Vision transformers have become essential for many medical image analysis tasks. They particular excel at capturing long-range interactions at the cost of a computational effort that grows quadratically with sequence length. This becomes even problematic for 3D problems, in particular for high-resolution diffusion models that require dozens of sampling steps. Flash attention successfully addressed the memory limitations of long sequences by optimising local memory access, but left the computational burden high. Quantising weights and activations for convolutions has been an active area of research for years, yielding completely binary networks that are however trained at higher precision and result in substantial performance drops. For transformers recent studies have been limited to quantising weights in linear layers or in an orthogonal research direction exploiting the potential of sparsity in self-attention scores. We present a novel scheme that not only enables a binary precision computation of the self-attention at inference time but even extends this to the training of transformers. To achieve differentiability we combine the bitwise Hamming distance with a learnable scalar query and key weighting. In theory this yields a 16-fold reduction in arithmetic operations and memory bandwidth - paving a way for more resource-efficient sequence modelling. We evaluate our model on three tasks with sequence lengths of N$>$1000, classification of images without patch-embedding, semantic 2D MRI segmentation and 3D diffusion models for inpainting and generative synthesise of high-resolution volumes. Our results demonstrate competitive performance and we provide an intuitive reasoning for the effectiveness of differentiable key-, query- weighting through Bernoulli sampling and high-dimensional interpolation.
