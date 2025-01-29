#required imports
import math
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.distributions as distributions
import torch.nn.functional as F
from tqdm.auto import tqdm,trange
import matplotlib.pyplot as plt
from IPython import display
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = str(3)
print(torch.cuda.get_device_name())
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

from codebook import *

import lpips
loss_fn_vgg = lpips.LPIPS(net='vgg').cuda() # closer to "traditional" perceptual loss, when used for optimization
import nibabel as nib
#brain = torch.from_numpy(nib.load('./brain3d.nii').get_fdata()).float().div(255).unsqueeze(0).unsqueeze(1)#[:,:,7:,7:]
#print(brain.shape,brain.max())

'''
reference: http://www.multisilicon.com/blog/a25332339.html
'''

class PixelShuffle3d(nn.Module):
    '''
    This class is a 3d version of pixelshuffle.
    '''
    def __init__(self, scale):
        '''
        :param scale: upsample scale
        '''
        super().__init__()
        self.scale = scale

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // self.scale ** 3

        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height, in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)
    
class PixelUnShuffle3d(nn.Module):
    '''
    This class is a 3d version of pixelshuffle.
    '''
    def __init__(self, scale):
        '''
        :param scale: upsample scale
        '''
        super().__init__()
        self.scale = scale

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels * self.scale ** 3

        out_depth = in_depth // self.scale
        out_height = in_height // self.scale
        out_width = in_width // self.scale

        input_view = input.contiguous().view(batch_size, channels, self.scale, self.scale, self.scale, out_depth, out_height, out_width)
        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)

#patch = torch.randn(4,1,16,16).cuda()
class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1a = []; self.l1b = []; self.l12 = []
        emb_dim = 256
        dim2 = 1024
        self.l0a = nn.Sequential(nn.Conv3d(emb_dim,dim2,1),nn.BatchNorm3d(dim2),nn.ReLU(),nn.Conv3d(dim2,dim2,1))
        for i in range(4):
            dim1 = 2**(10-i)#64
            dim2 = 2**(9-i)#128
            self.l1a.append(nn.Sequential(PixelShuffle3d(2),nn.Conv3d(dim1//8,dim1,1),nn.BatchNorm3d(dim1),nn.ReLU(True),nn.Conv3d(dim1,dim1//8,1),PixelUnShuffle3d(2)))
            self.l1b.append(nn.Sequential(nn.Conv3d(dim1,dim1,1),nn.BatchNorm3d(dim1),nn.ReLU(),nn.Conv3d(dim1,dim1,1)))
            self.l12.append(nn.ConvTranspose3d(dim1,dim2,2,stride=2))
            if(i>=2):
                self.l12.append(nn.Identity())            
                self.l1a.append(nn.Sequential(nn.Conv3d(dim2,dim1,3,padding=1),nn.BatchNorm3d(dim1),nn.ReLU(),nn.Conv3d(dim1,dim2,3,padding=1)))
                self.l1b.append(nn.Sequential(nn.Conv3d(dim2,dim1,1),nn.BatchNorm3d(dim1),nn.ReLU(),nn.Conv3d(dim1,dim2,1)))

        self.l1a = nn.Sequential(*self.l1a)
        self.l1b = nn.Sequential(*self.l1b)
        self.l12 = nn.Sequential(*self.l12)        
        self.l2b = nn.Sequential(nn.Conv3d(dim2,dim2,1),nn.BatchNorm3d(dim2),nn.ReLU(),nn.Conv3d(dim2,1,1))
    def forward(self, x):
        x = self.l0a(x)
        for j in range(len(self.l1a)):
            x = x + self.l1a[j](x)
            x = self.l12[j](x + self.l1b[j](x))
        x = self.l2b(x)
        return x
class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1a = []; self.l1b = []; self.l12 = []
        emb_dim = 256
        self.l0a = nn.Sequential(nn.Conv3d(1,64,1),nn.BatchNorm3d(64),nn.ReLU(),nn.Conv3d(64,64,1))
        for i in range(4):
            dim1 = 2**(i+6)#64
            dim2 = 2**(i+7)#128
            
            if(i<2):
                self.l12.append(nn.Identity())            
                self.l1a.append(nn.Sequential(nn.Conv3d(dim1,dim1*4,3,padding=1),nn.BatchNorm3d(dim1*4),nn.ReLU(),nn.Conv3d(dim1*4,dim1,3,padding=1)))
                self.l1b.append(nn.Sequential(nn.Conv3d(dim1,dim1*4,1),nn.BatchNorm3d(dim1*4),nn.ReLU(),nn.Conv3d(dim1*4,dim1,1)))

            self.l1a.append(nn.Sequential(PixelUnShuffle3d(2),nn.Conv3d(dim1*8,dim1,1),nn.BatchNorm3d(dim1),nn.ReLU(True),nn.Conv3d(dim1,dim1*8,1),PixelShuffle3d(2)))
            self.l1b.append(nn.Sequential(nn.Conv3d(dim1,dim1*4,1),nn.BatchNorm3d(dim1*4),nn.ReLU(),nn.Conv3d(dim1*4,dim1,1)))
            self.l12.append(nn.Conv3d(dim1,dim2,2,stride=2))
            
        self.l1a = nn.Sequential(*self.l1a)
        self.l1b = nn.Sequential(*self.l1b)
        self.l12 = nn.Sequential(*self.l12)       
        self.l2b = nn.Sequential(nn.Conv3d(dim2,dim2,1),nn.BatchNorm3d(dim2),nn.ReLU(),nn.Conv3d(dim2,emb_dim,1))
    def forward(self, x):
        x = self.l0a(x)
        for j in range(len(self.l1a)):
            x = x + self.l1a[j](x)

            x = self.l12[j](x + self.l1b[j](x))
        x = self.l2b(x)
        return x
patches_all = []
for i in trange(1,250):
    img1 = torch.from_numpy(nib.load('/home/jupyter-mattiastest/test_oasis/img/OASIS_0'+str(i).zfill(3)+'_0000.nii.gz').get_fdata()).half()

    patches1 = img1.unfold(0,32,16).unfold(1,32,16).unfold(2,32,16).reshape(-1,1,32,32,32)
    #patches = nn.Unfold((16,16,16),stride=(7,7,7))(brain)
    #print(patches1.shape)
    patches1 = patches1.flatten(1).mul(2).add(-1)
    #idx_valid = torch.nonzero((patches1.std(1)>0.15)&(patches1.std(1)<1)).squeeze()
    #print(idx_valid.shape,patches1.shape)
    patches_all.append(patches1.clone())
patches_all = torch.cat(patches_all,dim=0)

import sys
import time
t0 = time.time()

def normalise(x):
    return (x-x.min())/(x.max()-x.min())

#
dim_codebook = 32
num_codebook = 64
#vqae = CifarVQVAE(num_codebook,dim_codebook,'ema').cuda()
#codebook = EMACodebook(256,256).cuda()
#hamming = torch.compile(HammingQuantiser().cuda())#
encoder = (Encoder().cuda())#
decoder = (Decoder().cuda())#torch.compile
codebooks = nn.Sequential(*[EMACodebook(num_codebook,dim_codebook) for _ in range(8)]).cuda()

#decoder.load_state_dict(states[0])
#encoder.load_state_dict(states[1])

lr  = 3e-4
for rep in range(1,4):
    optimizer = torch.optim.Adam(list(codebooks.parameters())+list(decoder.parameters())+list(encoder.parameters()),lr=lr)
    num_iterations =8000
    schedulers = [torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0,\
                    total_iters=500),torch.optim.lr_scheduler.StepLR(optimizer,2000,0.5)]
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=schedulers, milestones=[500,])

    num_iterations =8000
    run_loss = torch.zeros(num_iterations,2)
    with tqdm(total=num_iterations, file=sys.stdout) as pbar:
        for i in range(num_iterations):

            optimizer.zero_grad()
            #idx = torch.randperm(50000,device='cuda')[:32].cpu()
            idx = torch.randperm(len(patches_all),device='cuda')[:64].cpu()#idx_valid[torch.randperm(len(idx_valid),device='cuda')[:128].cpu()]
            with torch.amp.autocast('cuda',dtype=torch.bfloat16):
                img = patches_all[idx].cuda().float().view(-1,1,32,32,32)#.repeat(1,3,1,1)
                
                #img = patches2d[idx].cuda().view(-1,1,32,32,32)#.repeat(1,3,1,1)
                #img = torch.from_numpy(cifar_data.data[idx]).permute(0,3,1,2).float().div(255).cuda()
                #encoding = autoencoder.encode(img)
                encoding = encoder(img)
                # Switch to channel last
                encoding = encoding.permute(0, 2, 3, 4, 1)

                #z = hamming(encoding.flatten(0,2)).view_as(encoding)
                z = torch.zeros_like(encoding)
                posterior_loss = 0
                for j in range(8):
                    quantized, _, codebook_metrics = codebooks[j].quantize(encoding[...,j*32:(j+1)*32])
                    posterior_loss += .125*codebook_metrics["loss_latent"]
                    z[...,j*32:(j+1)*32] = quantized
                
                quantized = z.permute(0, 4, 1, 2, 3)

                reconstructions = decoder(quantized)
                
                loss = F.mse_loss(reconstructions,img)
                reconstructions2d = torch.cat((reconstructions[...,16],reconstructions[...,16,:],reconstructions[...,15,:,:]),0)

                img2d = torch.cat((img[...,16],img[...,16,:],img[...,15,:,:]),0)

                loss_vgg = loss_fn_vgg(reconstructions2d.repeat(1,3,1,1),img2d.repeat(1,3,1,1))

                run_loss[i,0] = loss.item()*100
                run_loss[i,1] = loss_vgg.mean().item()*100
                loss += posterior_loss
                loss += loss_vgg.mean().mul(2)#posterior_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            

            str1 = f"iter: {i}, loss: {'%0.3f'%(run_loss[i-28:i-1,0].mean())}, vgg: {'%0.3f'%(run_loss[i-28:i-1,1].mean())}, runtime: {'%0.3f'%(time.time()-t0)} sec, GPU max/memory: {'%0.2f'%(torch.cuda.max_memory_allocated()*1e-9)} GByte"
            pbar.set_description(str1)
            pbar.update(1)

            if(i%200==199):
                
                f,ax = plt.subplots(3,3,figsize=(12,8))
                for k in range(9):
                    ax[k//3][k%3].imshow(torch.cat((normalise(img2d[k,0]),normalise(reconstructions2d[k,0]).float()),1).cpu().data,'gray');ax[k//3][k%3].axis('off')
                plt.savefig('results_reco/imgs_'+str(i)+'_'+str(rep)+'.png')
                plt.close()
                #plt.imshow(reconstructions[6].float().cpu().data.permute(1,2,0))
            if(i%2000==1999):
                torch.save([encoder.state_dict(),decoder.state_dict(),codebooks.state_dict(),run_loss.clone()], 'results_reco/weights_oasis_'+str(rep)+'.pth')
