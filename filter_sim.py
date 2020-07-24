import torchvision.models as models
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import torch

vgg = models.vgg16(pretrained=True)

def create_permutations(choices, slots):
    return np.array(list(product(choices, repeat=slots)))

layer_1 = vgg.features[0].weight.data.numpy() # 64, 3, 3, 3

filters = []
for layer in tqdm([0]): #, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28], desc='Getting Layer Filters'): # Conv2D layers
    for i in range(vgg.features[layer].out_channels): # filters
        for j in range(3): # rgb
            filters.append(vgg.features[layer].weight.data.numpy()[i, j, :, :]) # out, in, w, h (or h, w but who cares)
print("Filters got")
perms = create_permutations([-1,0,1], 9)
encodings = []
filters = [np.reshape(filter, 9) for filter in filters]

#filters = [torch.tensor(x).cuda() for x in filters]
#perms = [torch.tensor(x).cuda().float() for x in perms]

for filter in tqdm(filters, desc='Encoding Filters'):
    encoding = []
    for perm in perms:
       encoding.append(np.dot(filter, perm))
    norm = math.sqrt(np.dot(encoding, encoding))
    encodings.append([x/norm for x in encoding])
'''
data = torch.zeros([len(encodings), len(encodings)], dtype=torch.float32).cuda()
encodings = torch.stack([torch.tensor(x).cuda() for x in encodings]).cuda()


for i in tqdm(range(len(encodings)), desc='Comparing Encodings'):
    for j in range(len(encodings[0])):
        if i<=j:
            data[i][j] = torch.dot(encodings[i], encodings[j])
        else:
            pass
for i in tqdm(range(len(encodings)), desc='Finalizing Encodings'):
    for j in range(len(encodings[0])):
        if i>j:
            data[i][j] = data[j][i]
'''
data = []
encodings = torch.stack([torch.tensor(x).cuda() for x in encodings]).cuda()


for i in tqdm(range(len(encodings)), desc='Comparing Encodings'):
    xarr = [0]*len(encodings)
    for j in range(len(encodings)):
        if i<=j:
            xarr[j] = torch.dot(encodings[i], encodings[j]).item()
        else:
            pass
    data.append(xarr)
for i in tqdm(range(len(encodings)), desc='Finalizing Encodings'):
    for j in range(len(encodings)):
        if i>j:
            data[i][j] = data[j][i]

figure = plt.figure()
axes = figure.add_subplot(111)
caxes = axes.matshow(np.array(data), interpolation ='none')
figure.colorbar(caxes)