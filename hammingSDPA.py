import torch
from torch.nn import functional as F
import numpy as np
import warnings
from tqdm.auto import trange
warnings.filterwarnings("ignore", category=DeprecationWarning) 
#print(np.__version__)

#note: pytorch does not support uint64 so we are misusing int64
def packbits(x):
    x_b = torch.from_numpy(np.packbits((x>0).unflatten(-1,(-1,8)).byte()).view(np.int64))
    return x_b.view_as(x[...,:x.shape[-1]//64])
#note: bitwise_count counts bits of absolute value, hence view to uint64 necessary
def popcount(x,y):
    return torch.from_numpy(np.bitwise_count(x.numpy().view(np.uint64)^y.numpy().view(np.uint64)))
#numerically safe version of softmax 
def softmax(x):
    m = torch.max(x,-1,keepdim=True).values
    #m = np.max(a.numpy(),axis=1,keepdims=True)
    return torch.exp(x-m)/torch.sum(torch.exp(x-m),-1,keepdim=True)
class HammingAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q_,q_w,K_,k_w,V):
        D = 64
        S1 = popcount(Q_,K_.transpose(-2,-1)).to(V).sub(32).mul(-2)/ D**0.5
        S = S1 * q_w * k_w.transpose(-2,-1)
        P = softmax(S)
        O = np.matmul(P,V)
        ctx.save_for_backward(V,S1,P,q_w,k_w)
        return O
    @staticmethod
    def backward(ctx, dO):
        V,S1,P,q_w,k_w =  ctx.saved_tensors
        dV = np.matmul(P.transpose(-2,-1), dO)
        dP = np.matmul(dO, V.transpose(-2,-1))
        D_ = (P*dP).sum(-1,keepdim=True)
        dS =  P*(dP-D_)
        dq_w = (dS*S1*k_w.transpose(-2,-1)).sum(-1,keepdim=True)
        dk_w = (dS*S1*q_w).sum(-2).unsqueeze(-1)        
        return None,dq_w,None,dk_w,dV

class numpySDPA(torch.autograd.Function):
    @staticmethod
    def forward(ctx,Q,K,V):
        D = 64
        S = np.matmul(Q,K.transpose(-2,-1)) / D**.5
        P = softmax(S)
        O = np.matmul(P,V)
        ctx.save_for_backward(Q,K,V,S,P)
        return O
    @staticmethod
    def backward(ctx, dO):
        D = 64
        Q,K,V,S,P = ctx.saved_tensors
        dV = np.matmul(P.transpose(-2,-1),dO)
        dP = np.matmul(dO, V.transpose(-2,-1))
        D_ = (P*dP).sum(-1,keepdim=True)
        dS =  P*(dP-D_) #dSoftmax
        dQ = np.matmul(dS, K) / D**.5
        dK = np.matmul(dS.transpose(-2,-1), Q) / D**.5
        return dQ,dK,dV
sdpa_fp = numpySDPA.apply
sdpa = HammingAttention.apply
#create q,k,v
B,N,D = 16,1024,64
q = torch.sign(torch.randn(B,N,D)).detach()
q_w = torch.rand(B,N,1).requires_grad_()
k = torch.sign(torch.randn(B,N,D)).detach()
k_w = torch.rand(B,N,1).requires_grad_()
v = torch.randn(B,N,D).requires_grad_()
q_b,k_b = packbits(q).detach(),packbits(k).detach()

#q_b,k_b = packbits(q),packbits(k)
#h_dist = popcount(q_b^k_b.transpose(-2,-1))
#testing forward path
print('running correctness tests (on cpu)')
with torch.no_grad():
    out = sdpa(q_b,q_w,k_b,k_w,v)
    out1 = F.scaled_dot_product_attention(q*q_w,k*k_w,v)
    out2 = sdpa_fp(q*q_w,k*k_w,v)
    print('diff forward (%)',(out-out1).norm()/out1.norm(),' & ',(out2-out1).norm()/out1.norm(),)
q_w1 = q_w.clone().detach().requires_grad_()
k_w1 = k_w.clone().detach().requires_grad_()
v1 = v.clone().detach().requires_grad_()
q_w2 = q_w.clone().detach().requires_grad_()
k_w2 = k_w.clone().detach().requires_grad_()
v2 = v.clone().detach().requires_grad_()
    
out = sdpa(q_b,q_w,k_b,k_w,v)
out.square().mul(.5).sum().backward()
out1 = F.scaled_dot_product_attention(q*q_w1,k*k_w1,v1)
out1.square().mul(.5).sum().backward()
out2 = sdpa_fp(q*q_w2,k*k_w2,v2)
out2.square().mul(.5).sum().backward()
print('diff backward (%)',(q_w.grad-q_w1.grad).norm()/q_w1.grad.norm(),\
      (k_w.grad-k_w1.grad).norm()/k_w1.grad.norm(),(v.grad-v1.grad).norm()/v1.grad.norm())
print('& ',(q_w2.grad-q_w1.grad).norm()/q_w1.grad.norm(),\
      (k_w2.grad-k_w1.grad).norm()/k_w1.grad.norm(),(v2.grad-v1.grad).norm()/v1.grad.norm())
