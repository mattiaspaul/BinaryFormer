import torch
import os
import time,sys
import argparse
from transformers import ViTModel
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm,trange
from torcheval.metrics.functional import multiclass_f1_score

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

    def forward(self, x, y, z):
        x_s = x.size()[:-1] + (6,64//4)
        x_s4 = x.size()[:-1] + (6,64)#*4
        q,k,v = self.query(x),self.key(x), self.value(x)
        q_,k_ = self.activate(q).view(x_s4).transpose(2,1),self.activate(k).view(x_s4).transpose(2,1)
        v_ = v.view(x_s).transpose(2,1)
        y = F.scaled_dot_product_attention(q_,k_,v_).transpose(2,1).reshape(x[...,:384//4].size())
        return y,
class Attention1(nn.Module):
    def __init__(self,activate=nn.Identity()) -> None:
        super().__init__()
        self.activate = activate
        self.query = nn.Linear(384,6)
        self.key = nn.Linear(384,6)#*4
        self.value = nn.Linear(384,384//4)

    def forward(self, x, y, z):
        x_s = x.size()[:-1] + (6,64//4)
        x_s4 = x.size()[:-1] + (6,1)#*4
        q,k,v = self.query(x),self.key(x), self.value(x)
        q_,k_ = self.activate(q).view(x_s4).transpose(2,1),self.activate(k).view(x_s4).transpose(2,1)
        v_ = v.view(x_s).transpose(2,1)
        y = F.scaled_dot_product_attention(q_,k_,v_).transpose(2,1).reshape(x[...,:384//4].size())
        return y,

class BinaryAttention(nn.Module):
    def __init__(self,activate=nn.Identity()) -> None:
        super().__init__()
        self.activate = activate
        self.query = nn.Linear(384,384)
        self.key = nn.Linear(384,384)#*4
        self.value = nn.Linear(384,384//4)
        self.theta_Q = nn.Sequential(nn.Linear(384*2,192),nn.GELU(),nn.Linear(192,6))#,Tanh5(.5))#.cuda()
        self.theta_K = nn.Sequential(nn.Linear(384*2,192),nn.GELU(),nn.Linear(192,6))#,Tanh5(.5))#.cuda()

    def forward(self, x, y, z):
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
        return y,

class BinaryAttentionB(nn.Module):
    def __init__(self,activate) -> None:
        super().__init__()
        self.activate = activate
        self.query = nn.Linear(384,384)
        self.key = nn.Linear(384,384)#*4
        self.value = nn.Linear(384,384//4)
        #self.sdpa = myScaledDotProductAttention.apply
        #self.sdpa_l1 = myL1Attention.apply #not memory efficient

    def forward(self, x,y,z):
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
        y = F.scaled_dot_product_attention(qq,kk,v_,scale=1/x_s4[-1]**.5).unflatten(0,(-1,6)).transpose(2,1).reshape(x[...,:96].size())

        #y = self.sdpa(q_,k_,v_).transpose(2,1).reshape(x[...,:384//4].size())
        return y,

class BinaryAttentionG(nn.Module):
    def __init__(self,activate=nn.Identity()) -> None:
        super().__init__()
        self.activate = activate
        self.query = nn.Linear(384,384)
        self.key = nn.Linear(384,384)#*4
        self.value = nn.Linear(384,384//4)
        self.theta_Q = nn.Sequential(GroupedLinear(384*2,192),nn.GELU(),GroupedLinear(192,6))#,Tanh5(.5))#.cuda()
        self.theta_K = nn.Sequential(GroupedLinear(384*2,192),nn.GELU(),GroupedLinear(192,6))#,Tanh5(.5))#.cuda()

    def forward(self, x, y, z):
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
        return y,

def main(args):
    model = args.model
    gpu_num = args.gpu_num
    dataset = args.dataset
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    model1 = ViTModel.from_pretrained('facebook/dino-vits8').cuda()
    model1.apply(model1._init_weights)
    model1 = model1.encoder


    print(model,gpu_num,dataset)
    import torchvision

    if(dataset == 'mnist'):

        mnist = torchvision.datasets.MNIST('../')
        data = mnist.data.float().div(255)
        label = mnist.targets
        HW = 28; num_train = 60000; C=1

    if(dataset == 'cifar'):
        cifar = torchvision.datasets.CIFAR10('../')
        data = torch.from_numpy(cifar.data).float().div(255)
        label = torch.tensor(cifar.targets)
        HW = 32; num_train = 50000; C=3
    model2 = nn.Sequential(nn.Linear(C+2,384),nn.Linear(384,10)).cuda()


    #model1.apply(model1._init_weights)


    for i in range(len(model1.layer)):
        if(model == 'base'):
            if('b' in args.note):
                model1.layer[i].attention.attention = Attention1(nn.Identity()).cuda()
            else:
                model1.layer[i].attention.attention = Attention(nn.Identity()).cuda()
        elif(model == 'sign'):
            model1.layer[i].attention.attention = Attention(Sign5()).cuda()
        elif(model == 'sign_sg'):
            model1.layer[i].attention.attention = Attention(Sign0()).cuda()
        elif(model == 'hamming'):
            if('g' in args.note):
                model1.layer[i].attention.attention = BinaryAttentionG(Tanh5(.5)).cuda()
            elif('b' in args.note):
                model1.layer[i].attention.attention = BinaryAttentionB(Tanh5(.5)).cuda()
            else:  
                model1.layer[i].attention.attention = BinaryAttention(Tanh5(.5)).cuda()
        model1.layer[i].attention.output.dense = nn.Linear(384//4,384).cuda()
        
    model1 = torch.compile(model1)
    #training loop
    optimizer = torch.optim.Adam(list(model1.parameters())+list(model2.parameters()),lr=0.0001)
    num_iterations = 2500
    schedulers = [torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0,\
                    total_iters=300),torch.optim.lr_scheduler.StepLR(optimizer,6,0.98)]; 
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=schedulers, milestones=[300,])
    t0 = time.time()
    mesh = F.affine_grid(torch.eye(2,3).unsqueeze(0).repeat(32,1,1),(32,1,HW,HW))
    run_loss = torch.zeros(num_iterations,2)
    #run_loss = torch.zeros(num_iterations,2)
    with tqdm(total=num_iterations, file=sys.stdout) as pbar:
        for i in range(num_iterations):
            optimizer.zero_grad()
            idx = torch.randperm(num_train-10000,device='cuda')[:32].cpu()
            with torch.amp.autocast('cuda',dtype=torch.bfloat16):
                input = torch.cat((data[idx].reshape(-1,HW**2,C),mesh.view(-1,HW**2,2)),-1).cuda()
                output = model2[1](model1(model2[0](input))['last_hidden_state'].max(1).values)
                loss = nn.CrossEntropyLoss()(output,label[idx].cuda())
            loss.backward()
            optimizer.step()
            scheduler.step()
            run_loss[i,0] = loss.item()
            if(i%5==4):
                with torch.no_grad():
                    idx = torch.randperm(10000,device='cuda')[:32].cpu()+num_train-10000
                    with torch.amp.autocast('cuda',dtype=torch.bfloat16):
                        input = torch.cat((data[idx].reshape(-1,HW**2,C),mesh.view(-1,HW**2,2)),-1).cuda()
                        output = model2[1](model1(model2[0](input))['last_hidden_state'].max(1).values)

                run_loss[i-4:i+1,1] = (output.argmax(1).cpu()==label[idx]).float().mean().item()

            str1 = f"iter: {i}, acc: {'%0.3f'%(run_loss[i-28:i-1,1].mean())},  loss: {'%0.3f'%(run_loss[i-28:i-1,0].mean())}, runtime: {'%0.3f'%(time.time()-t0)} sec, GPU max/memory: {'%0.2f'%(torch.cuda.max_memory_allocated()*1e-9)} GByte"
            pbar.set_description(str1)
            pbar.update(1)
            if(i%1000==999):
                torch.save([model1.state_dict(),model2.state_dict(),run_loss],f'results_dino/{args.dataset}_{args.model}_{args.note}.pt')


    acc_val = []
    for j in trange(312):
        with torch.no_grad():
            idx = j*32+torch.arange(32)+num_train-10000
            with torch.amp.autocast('cuda',dtype=torch.bfloat16):
                input = torch.cat((data[idx].reshape(-1,HW**2,C),mesh.view(-1,HW**2,2)),-1).cuda()
                output = model2[1](model1(model2[0](input))['last_hidden_state'].max(1).values)

        acc_val.append(output.argmax(1).cpu()==label[idx])
    print('validation accuracy',torch.stack(acc_val).float().mean())
    torch.save([model1.state_dict(),model2.state_dict(),run_loss,acc_val],f'results_dino/{args.dataset}_{args.model}_{args.note}_val.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'finetune DINO on 2D segmentation, args: model, gpu_num, dataset')
    parser.add_argument('model', help='base, sign, hamming')
    parser.add_argument('gpu_num', help='usually 0-3')
    parser.add_argument('dataset', help='crossmoda, spine, chaos')
    parser.add_argument('note', help='experiment number')
    args = parser.parse_args()

    main(args)
