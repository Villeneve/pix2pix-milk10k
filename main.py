#%%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as tt, models as md
from torchinfo import summary

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.utils import *
from src.layers import *
from src.models import *

#%%
compose = tt.Compose([
    tt.Resize(142),
    tt.RandomCrop((128,128)),
    tt.ToTensor(),
    tt.Normalize(.5,.5),
])
ds = FolderLoad(compose)
loader = DataLoader(ds,40,True,num_workers=8)

#%%
device = torch.device('cuda:1')
gen = SimpleGenerator().to(device)
crit = SimpleCritic().to(device)
vgg = md.vgg16(weights=md.VGG16_Weights.DEFAULT).features[0:16]
vgg = vgg.to(device).eval()
setGrads(vgg,False)
summary(gen,(1,3,128,128),verbose=1)
# summary(crit,(1,3,256,256),verbose=1)
# summary(vgg,(1,3,224,224),verbose=1)
optG = torch.optim.Adam(
    params=gen.parameters(),
    lr=2e-4,
    betas=(.5,.999),
)
optC = torch.optim.Adam(
    params=crit.parameters(),
    lr=2e-4,
    betas=(.5,.999),
)

#%%
expAdvGen,expAdvCrit,expPercep = None,None,None
alpha_exp = .99
for epoch in range(1000+1):
    batchGraph = tqdm(loader)
    for derm,clin in batchGraph:
        # Data
        derm,clin = derm.to(device),clin.to(device)

        # Treino Critico
        setGrads(crit,True)
        fakeClin = gen(derm)
        trueLogits = crit(clin)
        fakeLogits = crit(fakeClin.detach())
        AdvCrit = (trueLogits-1).square().mean() + fakeLogits.square().mean()
        AdvCrit /= 2
        optC.zero_grad()
        AdvCrit.backward()
        optC.step()

        # Treino Gerador
        setGrads(crit,False)
        # Perceptual = nn.functional.l1_loss(
        #     vgg(prep_vgg(fakeClin)),
        #     vgg(prep_vgg(clin))
        # )
        Perceptual = nn.functional.l1_loss(fakeClin,clin)
        # optG.zero_grad()
        # Perceptual.backward(retain_graph=True)
        # PerceptualGradsNorm = torch.norm(torch.cat([p.grad.flatten() for p in gen.parameters()],0),2)
        # optG.step()

        AdvGen = (crit(fakeClin)-1).square().mean()
        # optG.zero_grad()
        # AdvGen.backward(retain_graph=True)
        # AdvGradsNorm = torch.norm(torch.cat([p.grad.flatten() for p in gen.parameters()],0),2)
        # optG.step()
        optG.zero_grad()
        gLoss = AdvGen + 10*Perceptual
        gLoss.backward()
        optG.step()

        # Plots
        if expAdvGen is None:
            expAdvGen = AdvGen.item()
            expAdvCrit = AdvCrit.item()
            expPercep = Perceptual.item()
        else:
            expAdvGen = alpha_exp*expAdvGen + (1-alpha_exp)*AdvGen.item()
            expAdvCrit = alpha_exp*expAdvCrit + (1-alpha_exp)*AdvCrit.item()
            expPercep = alpha_exp*expPercep + (1-alpha_exp)*Perceptual.item()
        dictLoss = {
            'Perceptual':f'{10*expPercep:.4f}',
            # 'MAEGradsNorm':f'{maeGradsNorm.item():.4f}',
            # 'AdvGradsNorm':f'{AdvGradsNorm.item():.4f}',
            'AdvGen':f'{expAdvGen:.4f}',
            'AdvCrit':f'{expAdvCrit:.4f}',
        }
        batchGraph.set_postfix(dictLoss)
        batchGraph.set_description(f'Epoch {epoch}')

    # Saves
    if epoch%5 == 0:
        torch.save(gen.state_dict(),'weights/gen.weights.pth')
        torch.save(crit.state_dict(),'weights/crit.weights.pth')
    with torch.inference_mode():
        derm,clin = next(iter(loader))
        derm,clin = derm.to(device),clin.to(device)
        imgs = gen(derm)
        imgs = toImage(imgs)
        k = 5
        fig,ax = plt.subplots(k,k,figsize=(8,8))
        ax = ax.ravel()
        for i in range(k**2):
            ax[i].imshow(imgs[i])
            ax[i].axis(False)
        plt.tight_layout(pad=0)
        plt.savefig('imgs.png')
        plt.close()
