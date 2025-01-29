#required imports
import math
import numpy as np
import os
import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.distributions as distributions
import torch.nn.functional as F
from tqdm.auto import tqdm,trange
import matplotlib.pyplot as plt

from torch.distributions import bernoulli as dists
def bilinaer_quantise(p1):
    p = p1.mul(.5)+.5
    #Bernoulli sampling
    b1 = dists.Bernoulli(probs=torch.clamp(p,0,1)).sample()
    b2 = dists.Bernoulli(probs=torch.clamp(p,0,1)).sample()
    #bilinear interpolation weights
    d1 = (p-b1).abs().mean(-1)+1e-12
    d2 = (p-b2).abs().mean(-1)+1e-12
    w1 = (d2)/(d1+d2)
    w2 = (d1)/(d1+d2)
    return b1,b2,w1,w2

class GroupedLinear(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.lin = nn.Conv2d(in_channel,out_channel,1,groups=6)
        nn.init.trunc_normal_(self.lin.weight,std=.02)
        nn.init.zeros_(self.lin.bias)

    def forward(self,x):
        return self.lin(x.unsqueeze(1).transpose(-1,1)).squeeze(-1).transpose(-1,1)
    
class Tanh5(nn.Module):
    def __init__(self,beta):
        super().__init__()
        self.beta = beta
    def forward(self,x):
        return 2*torch.tanh(self.beta*x)/min(self.beta,.75)
class mySign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):        
        ctx.save_for_backward(x)
        return (x>0).to(x).mul(2).add(-1)
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output*(x.abs()<1).to(grad_output).mul(2)

class Sign5(nn.Module):
    def __init__(self):
        super().__init__()
        self.mysign = mySign.apply
    def forward(self,x):
        return self.mysign(x)#4*torch.sign(x)
    
class mySign_SG(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):        
        ctx.save_for_backward(x)
        return (x>0).to(x).mul(4).add(-2)
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output*0#(x.abs()<1).to(grad_output).mul(2)

class Sign0(nn.Module):
    def __init__(self):
        super().__init__()
        self.mysign = mySign_SG.apply
    def forward(self,x):
        return self.mysign(x)#4*torch.sign(x)


class Decoder(nn.Module):
    def __init__(self,max_label) -> None:
        super().__init__()
        self.l1a = []; self.l1b = []; self.l12 = []
        emb_dim = 384
        dim2 = 1024
        self.l0a = nn.Sequential(nn.Conv2d(emb_dim,dim2,1),nn.BatchNorm2d(dim2),nn.ReLU(),nn.Conv2d(dim2,dim2,1))
        for i in range(3):
            dim1 = 2**(10-i)#64
            dim2 = 2**(9-i)#128
            self.l1a.append(nn.Sequential(nn.PixelShuffle(2),nn.Conv2d(dim1//4,dim1,1),nn.BatchNorm2d(dim1),nn.ReLU(True),nn.Conv2d(dim1,dim1//4,1),nn.PixelUnshuffle(2)))
            self.l1b.append(nn.Sequential(nn.Conv2d(dim1,dim1,1),nn.BatchNorm2d(dim1),nn.ReLU(),nn.Conv2d(dim1,dim1,1)))
            self.l12.append(nn.ConvTranspose2d(dim1,dim2,2,stride=2))
            if(i>=2):
                self.l12.append(nn.Identity())            
                self.l1a.append(nn.Sequential(nn.Conv2d(dim2,dim1,3,padding=1),nn.BatchNorm2d(dim1),nn.ReLU(),nn.Conv2d(dim1,dim2,3,padding=1)))
                self.l1b.append(nn.Sequential(nn.Conv2d(dim2,dim1,1),nn.BatchNorm2d(dim1),nn.ReLU(),nn.Conv2d(dim1,dim2,1)))

        self.l1a = nn.Sequential(*self.l1a)
        self.l1b = nn.Sequential(*self.l1b)
        self.l12 = nn.Sequential(*self.l12)        
        self.l2b = nn.Sequential(nn.Conv2d(dim2,dim2,1),nn.BatchNorm2d(dim2),nn.ReLU(),nn.Conv2d(dim2,max_label,1))
    def forward(self, x):
        x = self.l0a(x)
        for j in range(len(self.l1a)):
            x = x + self.l1a[j](x)
            x = self.l12[j](x + self.l1b[j](x))
        x = self.l2b(x)
        return x
class Attention(nn.Module):
    def __init__(self,activate=nn.Identity()) -> None:
        super().__init__()
        self.activate = activate
        self.query = nn.Linear(384,384)
        self.key = nn.Linear(384,384)#*4
        self.value = nn.Linear(384,384//4)
        self.dense = nn.Linear(384//4,384)

    def forward(self, x, y, z,attn_mask,key_padding_mask, need_weights, is_causal):
        x_s = x.size()[:-1] + (6,64//4)
        x_s4 = x.size()[:-1] + (6,64)#*4
        q,k,v = self.query(x),self.key(x), self.value(x)
        q_,k_ = self.activate(q).view(x_s4).transpose(2,1),self.activate(k).view(x_s4).transpose(2,1)
        v_ = v.view(x_s).transpose(2,1)
        y = F.scaled_dot_product_attention(q_,k_,v_).transpose(2,1).reshape(x[...,:384//4].size())
        return self.dense(y),
class Attention1(nn.Module):
    def __init__(self,activate=nn.Identity()) -> None:
        super().__init__()
        self.activate = activate
        self.query = nn.Linear(384,6)
        self.key = nn.Linear(384,6)#*4
        self.value = nn.Linear(384,384//4)
        self.dense = nn.Linear(384//4,384)

    def forward(self, x, y, z,attn_mask,key_padding_mask, need_weights, is_causal):
        x_s = x.size()[:-1] + (6,64//4) 
        x_s4 = x.size()[:-1] + (6,1)#*4
        q,k,v = self.query(x),self.key(x), self.value(x)
        q_,k_ = self.activate(q).view(x_s4).transpose(2,1),self.activate(k).view(x_s4).transpose(2,1)
        v_ = v.view(x_s).transpose(2,1)
        y = F.scaled_dot_product_attention(q_,k_,v_).transpose(2,1).reshape(x[...,:384//4].size())
        return self.dense(y),


class BinaryAttention(nn.Module):
    def __init__(self,activate=nn.Identity()) -> None:
        super().__init__()
        self.activate = activate
        self.query = nn.Linear(384,384)
        self.key = nn.Linear(384,384)#*4
        self.value = nn.Linear(384,384//4)
        self.theta_Q = nn.Sequential(nn.Linear(384*2,192),nn.GELU(),nn.Linear(192,6))#,Tanh5(.5))#.cuda()
        self.theta_K = nn.Sequential(nn.Linear(384*2,192),nn.GELU(),nn.Linear(192,6))#,Tanh5(.5))#.cuda()
        self.dense = nn.Linear(384//4,384)

    def forward(self, x, y, z,attn_mask,key_padding_mask, need_weights, is_causal):
        x_s = x.size()[:-1] + (6,64//4)
        x_s4 = x.size()[:-1] + (6,64)#*4
        x_s1 = x.size()[:-1] + (6,1)

        q,k,v = self.query(x),self.key(x), self.value(x)
        q_,k_ = self.activate(q),self.activate(k)
        Q_b,K_b = torch.sign(q).detach(),torch.sign(k).detach() #no grad necessary
        q_w = self.theta_Q(torch.cat((q_,Q_b),-1)) #learned scalar weight/head
        k_w = self.theta_K(torch.cat((k_,K_b),-1)) # " = "
        Q_ = Q_b.view(x_s4).transpose(2,1) * q_w.view(x_s1).transpose(2,1)
        K_ = K_b.view(x_s4).transpose(2,1) * k_w.view(x_s1).transpose(2,1)
        v_ = v.view(x_s).transpose(2,1)
        y = F.scaled_dot_product_attention(Q_*2,K_*2,v_).transpose(2,1).reshape(x[...,:384//4].size())
        return self.dense(y),

class BinaryAttentionB(nn.Module):
    def __init__(self,activate) -> None:
        super().__init__()
        self.activate = activate
        self.query = nn.Linear(384,384)
        self.key = nn.Linear(384,384)#*4
        self.value = nn.Linear(384,384//4)
        #self.sdpa = myScaledDotProductAttention.apply
        #self.sdpa_l1 = myL1Attention.apply #not memory efficient
        self.dense = nn.Linear(384//4,384)

    def forward(self, x, y, z,attn_mask,key_padding_mask, need_weights, is_causal):
        x_s = x.size()[:-1] + (6,16)
        x_s4 = x.size()[:-1] + (6,64)#*4
        q,k,v = self.query(x),self.key(x), self.value(x)
        q_,k_ = self.activate(q.view(x_s4).transpose(2,1).flatten(0,1)),self.activate(k.view(x_s4).transpose(2,1).flatten(0,1))
        v_ = v.view(x_s).transpose(2,1).flatten(0,1)
        q1,q2,w_q1,w_q2 = bilinaer_quantise(q_)
        qw1 = q1.mul(2).sub(1)*w_q1.unsqueeze(-1); qw2 = q2.mul(2).sub(1)*w_q2.unsqueeze(-1)
        k1,k2,w_k1,w_k2 = bilinaer_quantise(k_)
        kw1 = k1.mul(2).sub(1)*w_k1.unsqueeze(-1); kw2 = k2.mul(2).sub(1)*w_k2.unsqueeze(-1)

        qq = torch.cat((qw1,qw2,qw1,qw2),-1)
        kk = torch.cat((kw1,kw1,kw2,kw2),-1)
        y = F.scaled_dot_product_attention(qq,kk,v_,scale=1/x_s4[-1]**.5).unflatten(0,(-1,6)).reshape(x[...,:96].size())

        #y = self.sdpa(q_,k_,v_).transpose(2,1).reshape(x[...,:384//4].size())
        return self.dense(y),

class BinaryAttentionG(nn.Module):
    def __init__(self,activate=nn.Identity()) -> None:
        super().__init__()
        self.activate = activate
        self.query = nn.Linear(384,384)
        self.key = nn.Linear(384,384)#*4
        self.value = nn.Linear(384,384//4)
        self.theta_Q = nn.Sequential(GroupedLinear(384*2,192),nn.GELU(),GroupedLinear(192,6))#,Tanh5(.5))#.cuda()
        self.theta_K = nn.Sequential(GroupedLinear(384*2,192),nn.GELU(),GroupedLinear(192,6))#,Tanh5(.5))#.cuda()
        self.dense = nn.Linear(384//4,384)
    def forward(self, x, y, z,attn_mask,key_padding_mask, need_weights, is_causal):
        x_s = x.size()[:-1] + (6,64//4)
        x_s4 = x.size()[:-1] + (6,64)#*4
        x_s1 = x.size()[:-1] + (6,1)

        q,k,v = self.query(x),self.key(x), self.value(x)
        q_,k_ = self.activate(q),self.activate(k)
        Q_b,K_b = torch.sign(q).detach(),torch.sign(k).detach() #no grad necessary
        q_w = self.theta_Q(torch.cat((q_,Q_b),-1)) #learned scalar weight/head
        k_w = self.theta_K(torch.cat((k_,K_b),-1)) # " = "
        Q_ = Q_b.view(x_s4).transpose(2,1) * q_w.view(x_s1).transpose(2,1)
        K_ = K_b.view(x_s4).transpose(2,1) * k_w.view(x_s1).transpose(2,1)
        v_ = v.view(x_s).transpose(2,1)
        y = F.scaled_dot_product_attention(Q_*2,K_*2,v_).transpose(2,1).reshape(x[...,:384//4].size())
        return self.dense(y),

class Transformer(nn.Module):
    def __init__(self,attn=Attention(),N=16*24*20) -> None:
        super().__init__()
        self.token_embedding = nn.Sequential(*[nn.Embedding(65, 48) for _ in range(8)])
#        self.position_embedding = nn.Embedding(10*14*12, 384)
        self.N = N # 16*24*20 or 10*14*12
        self.position_embedding = nn.Embedding(N, 384)
        self.layer_norm = nn.LayerNorm(384)
        
        # provide initialisation 
        for i in range(8):
            self.token_embedding[i].weight.data.normal_(.0,.02)
        self.position_embedding.weight.data.normal_(.0,.02)
        self.layer_norm.bias.data.zero_()
        self.layer_norm.weight.data.fill_(1.0)
        self.head = nn.Linear(384, 8*64, bias=False)
        self.head.weight.data.normal_(.0,.02)
        transformer = []
        for _ in range(8):
            t = nn.TransformerEncoderLayer(384, 6, dim_feedforward=4*384,\
                                            batch_first=True, activation=nn.GELU())
            t.self_attn = attn
            transformer.append(t)
        self.transformer = nn.Sequential(*transformer)
#        self.transformer = nn.Sequential(*[SelfAttentionMLP1(Tanh5(.5)) for _ in range(8)])

    def forward(self, q):
#        x = torch.zeros(q.shape[0],10*14*12,384,device='cuda')#,self.token_embedding(q.view(-1,10*14*12))
        x = torch.zeros(q.shape[0],self.N,384,device='cuda')#,self.token_embedding(q.view(-1,10*14*12))
        for i in range(8):
            x[:,:,i*48:(i+1)*48] = self.token_embedding[i](q[:,i].long())
        x += self.position_embedding.weight.unsqueeze(0)
        x = self.transformer(x)
        return self.head(self.layer_norm(x))
def weighted_elbo_loss(x_logits,target,t):
    cross_entropy_loss = 0
    for i in range(8):
        cross_entropy_loss += .125*F.cross_entropy(x_logits[:,i*64:(i+1)*64], target[:,i], ignore_index=-1, reduction='none').sum(1)

    #cross_entropy_loss = F.cross_entropy(x_logits, target, ignore_index=-1, reduction='none').sum(1)
    weight = (1 - t) #lower weight for earlier (more difficult) time points
    loss = weight * cross_entropy_loss
    loss = loss / (float(torch.log(torch.tensor([2]))) * x_logits.shape[1:].numel())
    return loss.mean()


def optim_warmup(lr_base, step, optim, warmup_iters):
    lr = lr_base * float(step) / warmup_iters
    for param_group in optim.param_groups:
        param_group['lr'] = lr

def main(args):
    model = args.model
    gpu_num = args.gpu_num
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True


    if(args.note == '16'):
        q_all,_ = torch.load('q_all_oasis_3.pth')
        H1,W1,D1,div,B = 10, 14, 12, 2, 12
    else:
        _,q_all = torch.load('q_all_oasis_3_64.pth')
        H1,W1,D1,div,B  = 16, 24, 20, 4, 8
    N = H1*W1*D1
    if model == 'base':
        denoise_fn = Transformer(Attention(),N).cuda()
    elif model == 'base1':
        denoise_fn = Transformer(Attention1(),N).cuda()
    elif model == 'sign':
        denoise_fn = Transformer(Attention(Sign5()),N).cuda()
    elif model == 'sign_sg':
        denoise_fn = Transformer(Attention(Sign0()),N).cuda()
    elif model == 'hammingG':
        denoise_fn = Transformer(BinaryAttentionG(Tanh5(.5)),N).cuda()
    elif model == 'hamming1':
        denoise_fn = Transformer(BinaryAttention(Tanh5(.5)),N).cuda()
    elif model == 'hammingB':
        denoise_fn = Transformer(BinaryAttentionB(Tanh5(.5)),N).cuda()



    #with torch.no_grad():
        ##sanity check for shapes:
    #    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    #        x_logits = denoise_fn(q_all[:4].cuda().long().flatten(2)).permute(0,2,1)
    #        print(x_logits.shape) #should be of shape 12 x 1024 x 256
    #provided

            
    lr_base = 1e-4
    warmup_iters = 500
    num_timesteps = 256
    mask_id = 64
    num_iterations = 21000

    #initially we use just 499 iterations (to prepare Task 3)
   # num_iterations = 7500#17500#499

    #f,ax = plt.subplots(1,2,figsize=(14,6))
    t0 = time.time()
    optimizer = torch.optim.Adam(denoise_fn.parameters(),lr=lr_base)
    run_loss = torch.zeros(num_iterations)
    #pbar = tqdm(iterable=range(num_iterations), leave=True)
    with tqdm(total=num_iterations, file=sys.stdout) as pbar:
        for i in range(num_iterations):

    #for i in pbar:
            denoise_fn.train()

            if i <= warmup_iters:
                optim_warmup(lr_base, i, optimizer, warmup_iters)


            optimizer.zero_grad()

            ##TODO
        #    idx = torch.randperm(len(q_all)-50)[:12]
            idx = torch.randperm(len(q_all)-50)[:B]
        #    x_0 = q_all[idx].view(len(idx),8,-1).cuda()
            x_0 = q_all[idx].view(len(idx),8,-1).cuda().long()
            # sample random time point for each batch element
            t = torch.randint(1, num_timesteps+1, (x_0.shape[0],)).to(x_0).float() / num_timesteps
            # create and apply random mask
        #    structured_mask = F.interpolate(torch.rand_like(x_0[:,:1,::8].float()).reshape(12,1,5,7,6),scale_factor=2,mode='nearest').flatten(2)
            structured_mask = F.interpolate(torch.rand_like(x_0[:,:1,::div**3].float()).reshape(B,1,H1//div,W1//div,D1//div),scale_factor=div,mode='nearest').flatten(2)
            mask = structured_mask.repeat(1,8,1) < (t.float()).unsqueeze(-1).unsqueeze(-1)

            #mask = torch.rand_like(x_0[:,:1].float()).repeat(1,8,1) < (t.float()).unsqueeze(-1).unsqueeze(-1)
            # replace masked tokens with the undefined ID (1024)
            x_t = torch.where(mask,mask_id,x_0.clone())
            # ground-truth with unchangable tokens set to ignore 
            target = torch.where(mask,x_0.clone(),-1) 
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # perform one denoising step 
                x_logits = denoise_fn(x_t).permute(0,2,1)

                #GIVEN
                loss = weighted_elbo_loss(x_logits,target,t)
            loss.backward()
            optimizer.step() #TODO fp16 (for T4)

            run_loss[i] = loss.item()*100

            str1 = f"iter: {i}, loss: {'%0.3f'%(run_loss[i-28:i-1].mean())}, runtime: {'%0.3f'%(time.time()-t0)} sec, GPU max/memory: {'%0.2f'%(torch.cuda.max_memory_allocated()*1e-9)} GByte"
            pbar.set_description(str1)
            pbar.update(1)
            if(i == 10):
                torch.save(denoise_fn.state_dict(),f'results_sampler/denoise_fn_{model}_{args.note}_{i}.pth')
                torch.save(run_loss,f'results_sampler/run_loss_{model}_{args.note}_{i}.pth')

            if(i%3500 == 3499):
                torch.save(denoise_fn.state_dict(),f'results_sampler/denoise_fn_{model}_{args.note}_{i}.pth')
                torch.save(run_loss,f'results_sampler/run_loss_{model}_{args.note}_{i}.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'train 3D vq diffusion model, args: model, gpu_num, note')
    parser.add_argument('model', help='base, sign, hamming')
    parser.add_argument('gpu_num', help='usually 0-3')
    parser.add_argument('note', help='experiment number')
    args = parser.parse_args()

    main(args)
