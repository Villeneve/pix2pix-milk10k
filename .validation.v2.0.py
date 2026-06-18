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
    tt.Resize(128),
    tt.CenterCrop((128,128)),
    tt.ToTensor(),
    tt.Normalize(.5,.5),
])
# compose = tt.Compose([
#     tt.Resize(256),
#     tt.RandomCrop((256,256)),
#     tt.ToTensor(),
#     tt.Normalize(.5,.5),
# ])
ds = FolderLoad(compose)
loader = DataLoader(ds,32,False,num_workers=8)

#%%
device = torch.device('cuda:0')
gen = SimpleGenerator().to(device)
# summary(gen,(1,3,128,128),verbose=1)
# gen.load_state_dict(torch.load('weights/gen.weights.pth'))
# gen.load_state_dict(torch.load('weights/gen.vgg16.v1.0.10L2.weights.pth'))


#% Select image
# n = 1677
# n = 167
# n = 853
# n = 5101
# n = 5035
n = random.randint(0,len(ds))
print(n)
dermt,clint = ds[n]
dermt,clint = dermt.unsqueeze(0),clint.unsqueeze(0)
dermt = dermt[0:1]
clint = clint[0:1]  
dermt,clint = dermt.to(device),clint.to(device)

weights = {
    'vgg16.10L2':'weights/gen.vgg16.v2.0.10L2.weights.pth',
    'vgg16.10L9':'weights/gen.vgg16.v2.0.10L9.weights.pth',
    # 'weights/gen.vgg16.v1.0.10L2.weights.pth',
    # 'weights/gen.vgg16.v1.0.L2.weights.pth',
    # 'weights/gen.vgg16.v1.0.L9.weights.pth',
    # 'weights/gen.vgg16.v1.0.L16.weights.pth'
}
names = [
    
    
    # 'vgg16.L2',
    # 'vgg16.L9',
    # 'vgg16.L16',
]
n = 4
fig, ax = plt.subplots(1,2+len(weights),figsize=((len(names)+2)*n,n),dpi=300)
ax = ax.ravel()
derm = toImage(dermt)[0]
clin = toImage(clint)[0]
ax[0].imshow(derm)
ax[0].set_title('Input',fontsize=24)
ax[0].axis(False)
ax[1].imshow(clin)
ax[1].set_title('Ground',fontsize=24)
ax[1].axis(False)
with torch.inference_mode():
    for i,(name,weight) in enumerate(weights.items()):
        gen.load_state_dict(torch.load(weight))
        img = gen(dermt)
        img = toImage(img)[0]
        ax[i+2].imshow(img)
        ax[i+2].set_title(name,fontsize=24)
        ax[i+2].axis(False)
plt.tight_layout(pad=.05)
plt.savefig('gen.png',bbox_inches='tight')
plt.close()
# %%
