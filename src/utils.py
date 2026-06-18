#%%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import models as md
from PIL import Image
import os
import random
import pandas as pd

#%%
class FolderLoad(Dataset):
    def __init__(self, transform=None):
        super().__init__()
        self.transform = transform
        self.imagePath = '/storage/SSD1/.data/milk10k/images/'
        df = pd.read_csv('/storage/SSD1/.data/milk10k/supplements/training_input.csv')
        df = df[['lesion_id','image_type','isic_id']]
        df_derm = df[df['image_type']=='dermoscopic']
        df_clin = df[df['image_type']=='clinical: close-up']
        self.df = pd.merge(df_derm,df_clin,on='lesion_id')

    def __getitem__(self, index):
        derm = self.imagePath + self.df.iloc[index]['isic_id_x'] + '.jpg'
        clin = self.imagePath + self.df.iloc[index]['isic_id_y'] + '.jpg'
        derm = Image.open(derm)
        clin = Image.open(clin)

        if self.transform is not None:
            return self.transform(derm),self.transform(clin)
        return derm,clin

    def __len__(self):
        return len(self.df)

#%%
def toImage(x:torch.Tensor):
    if len(x.shape) == 3:
        x = x.detach().permute(1,2,0).cpu().numpy()
        x *= 127.5
        x += 127.5
        x = x.astype(np.uint8)
        return x
    x = x.detach().permute(0,2,3,1).cpu().numpy()
    x *= 127.5
    x += 127.5
    x = x.astype(np.uint8)
    return x

def plotResult(gen:nn.Module,loader):
    with torch.no_grad():
        gen.eval()
        horses,_ = next(iter(loader))
        horses = horses.to(next(gen.parameters()).device)
        fake_horses = gen(horses)
        imgs = torch.cat([horses,fake_horses],dim=0)
        width = imgs.size(0)
        fig,ax = plt.subplots(2,width//2,figsize=(width//2,2))
        ax = ax.ravel()
        for i in range(width):
            ax[i].imshow(toImage(imgs[i]))
            ax[i].axis(False)
        plt.tight_layout(pad=0)
        plt.savefig('result.png')
        plt.close()
    gen.train()

def setGrads(model:nn.Module,value:bool):
    for p in model.parameters():
        p.requires_grad_(value)

import torch.nn.functional as F

def patch_nce_loss(
    feat_q: torch.Tensor,
    feat_k: torch.Tensor,
    temperature: float = 0.07,
    detach_key: bool = True,
):
    if detach_key:
        feat_k = feat_k.detach()

    B, S, C = feat_q.shape

    l_pos = torch.bmm(
        feat_q.reshape(B * S, 1, C),
        feat_k.reshape(B * S, C, 1)
    ).reshape(B, S, 1)

    l_neg = torch.bmm(
        feat_q,
        feat_k.transpose(1, 2)
    )

    diagonal = torch.eye(S, device=feat_q.device, dtype=torch.bool)[None, :, :]
    l_neg = l_neg.masked_fill(diagonal, -10.0)

    logits = torch.cat([l_pos, l_neg], dim=2)
    logits = logits / temperature
    logits = logits.reshape(B * S, 1 + S)

    labels = torch.zeros(B * S, dtype=torch.long, device=feat_q.device)

    return F.cross_entropy(logits, labels)

def lr_lambda(epoch: int) -> float:
    # epoch é a época atual (começa em 0)
    # retorna um MULTIPLICADOR, não o LR diretamente
    if epoch < 100:
        return 1.0                        # LR = 2e-4 × 1.0 = 2e-4 (sem mudança)
    else:
        return 1.0 - (epoch - 100) / 900 # decai linearmente até 0 ao fim de 1000 épocas
    
def multilayer_patch_nce_loss(
    feats_q,
    feats_k,
    netF,
    num_patches=256,
    temperature=0.07,
):
    feats_q, patch_ids = netF(
        feats_q,
        patch_ids=None,
        num_patches=num_patches
    )

    feats_k, _ = netF(
        feats_k,
        patch_ids=patch_ids,
        num_patches=num_patches
    )

    total_loss = 0.0

    for feat_q, feat_k in zip(feats_q, feats_k):
        total_loss = total_loss + patch_nce_loss(
            feat_q,
            feat_k,
            temperature=temperature
        )

    return total_loss / len(feats_q)

def prep_vgg(x):
    x = ((x + 1) / 2)
    x = x.clamp(0, 1)
    x = md.VGG16_Weights.DEFAULT.transforms()(x)
    return x

def covariance(x: torch.Tensor):
    mean = x.mean(0,keepdim=True)
    x -= mean
    cov = (x.T@x) / (x.size(0)-1)
    return cov

def matrix_sqrt_psd(mat, eps=1e-6):
    """
    Raiz quadrada de matriz simétrica positiva semi-definida.
    Usa decomposição em autovalores.
    """
    mat = (mat + mat.T) / 2  # força simetria numérica

    eigvals, eigvecs = torch.linalg.eigh(mat)
    eigvals = torch.clamp(eigvals, min=eps)

    sqrt_eigvals = torch.sqrt(eigvals)

    sqrt_mat = eigvecs @ torch.diag(sqrt_eigvals) @ eigvecs.T
    return sqrt_mat


def fid(real_features, fake_features, eps=1e-6):
    """
    real_features: tensor (N, D)
    fake_features: tensor (N, D)
    retorna escalar FID
    """
    real_features = real_features.float()
    fake_features = fake_features.float()

    mu_r = real_features.mean(dim=0)
    mu_g = fake_features.mean(dim=0)

    sigma_r = covariance(real_features)
    sigma_g = covariance(fake_features)

    diff = mu_r - mu_g
    mean_term = diff @ diff

    # Forma numericamente mais estável:
    # Tr((Σr Σg)^1/2) = Tr((Σr^1/2 Σg Σr^1/2)^1/2)
    dim = sigma_r.shape[0]
    eye = torch.eye(dim, device=sigma_r.device, dtype=sigma_r.dtype)

    sigma_r = sigma_r + eps * eye
    sigma_g = sigma_g + eps * eye

    sigma_r_sqrt = matrix_sqrt_psd(sigma_r, eps=eps)
    middle = sigma_r_sqrt @ sigma_g @ sigma_r_sqrt
    covmean = matrix_sqrt_psd(middle, eps=eps)

    cov_term = torch.trace(sigma_r + sigma_g - 2 * covmean)

    fid_value = mean_term + cov_term

    return torch.clamp(fid_value, min=0)