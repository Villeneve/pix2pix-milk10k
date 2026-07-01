''''
IFPB - Instituto Federal da Paraíba
João Pessoa - Junho 2026
Author - Me. Villeneve de Oliveira Soares

Obs: Cada treinamento levou 1000 épocas com 131 atualizações cada época.
'''

#%%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as tt, models as md
from torchinfo import summary

import matplotlib.pyplot as plt
import numpy as np
import shutil
from datetime import datetime
from tqdm import tqdm


from src.utils import *
from src.layers import *
from src.models import *

#%%
compose = tt.Compose([
    tt.Resize(128),
    tt.CenterCrop((128,128)),
    tt.ToTensor(),
    # tt.Normalize(127.5,127.5),
    tt.Normalize(.5,.5),
])
deNormalize = tt.Compose([
    tt.Normalize(-1,2),
])
# resize_crop = tt.Compose([
#     tt.Resize(128),
#     tt.CenterCrop((128,128))
# ])
# resize_fn = tt.Resize((224,224))
# compose = tt.Compose([
#     tt.Resize(256),
#     tt.RandomCrop((256,256)),
#     tt.ToTensor(),
#     tt.Normalize(.5,.5),
# ])
# ds = FolderLoad(tt.PILToTensor())
ds = FolderLoad(compose)
loader = DataLoader(ds,512,True,num_workers=8)

device = torch.device('cuda:0')
inception = md.inception_v3(weights=md.Inception_V3_Weights.DEFAULT).to(device)
inception.fc = nn.Identity()
preprocess = md.Inception_V3_Weights.DEFAULT.transforms()
summary(inception,(1,3,299,299),verbose=1)

weights = {
    '10MAE':'weights/gen.MAE.v1.0.10.weights.pth',
    'vgg16.L2':'weights/gen.vgg16.v1.0.L2.weights.pth',
    'vgg16.10L2':'weights/gen.vgg16.v1.0.10L2.weights.pth',
    'vgg16.L9':'weights/gen.vgg16.v1.0.L9.weights.pth',
    'vgg16.10L9':'weights/gen.vgg16.v1.0.10L9.weights.pth',
    'vgg16.L16':'weights/gen.vgg16.v1.0.L16.weights.pth',
    'vgg16.10L16':'weights/gen.weights.pth'
}

gen = SimpleGenerator().to(device)

#%%
# for (name,path) in weights.items():
#     gen.load_state_dict(torch.load(path))
#     # print(name)

#     gen.eval()
#     inception.eval()
#     fakeFeatures,trueFeatures = [],[]
#     for batch,_ in tqdm(loader,leave=False):
#         with torch.inference_mode():

#             batch = batch.to(device)
#             batch = compose(batch.float())
#             imgsFakes = gen(batch)

#             imgsFakes = preprocess(deNormalize(imgsFakes))
#             fakeFeatures_ = inception(imgsFakes)
#             trueFeatures_ = inception(preprocess(deNormalize(batch)))

#             fakeFeatures.append(fakeFeatures_)
#             trueFeatures.append(trueFeatures_)

#     fakeFeatures = torch.cat(fakeFeatures,0)
#     trueFeatures = torch.cat(trueFeatures,0)
#     print(f'{name}: {fid(trueFeatures,fakeFeatures)}')

        


#%%
gen = SimpleGenerator().to(device)
# gen.load_state_dict(torch.load('weights/gen.weights.pth'))
# gen.load_state_dict(torch.load('weights/gen.vgg16.v1.0.10L2.weights.pth'))


#% Select image
# n = 1677
# n = 167
# n = 853
# n = 5101
# n = 5035
for n in range(len(ds)):
    # n = random.randint(0,len(ds))
    print(n)
    dermt,clint = ds[n]
    dermt,clint = dermt.unsqueeze(0),clint.unsqueeze(0)
    dermt = dermt[0:1]
    clint = clint[0:1]
    dermt,clint = dermt.to(device),clint.to(device)

    weights = {
        '10MAE':'weights/gen.MAE.v1.0.10.weights.pth',
        'vgg16.L2':'weights/gen.vgg16.v1.0.L2.weights.pth',
        'vgg16.10L2':'weights/gen.vgg16.v1.0.10L2.weights.pth',
        'vgg16.L9':'weights/gen.vgg16.v1.0.L9.weights.pth',
        'vgg16.10L9':'weights/gen.vgg16.v1.0.10L9.weights.pth',
        'vgg16.L16':'weights/gen.vgg16.v1.0.L16.weights.pth',
        'vgg16.10L16':'weights/gen.weights2.pth'
    }

    n = 4
    fig, ax = plt.subplots(1,2+len(weights),figsize=((2+len(weights))*n,n),dpi=300)
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
    plt.show()
    choice = input('Salvar? ')
    if choice != 'n':
        shutil.copy2('./gen.png',f'examples/{choice}/{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    if choice == 'q':
        raise SystemExit

