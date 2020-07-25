import torchvision.models as models
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import torch
from util import *


linSort = False
hierSort = True
vis = True
randMat = False
method = "randomMatrixGeneration" if randMat else "randomNumberPermutation"

vgg = models.vgg16(pretrained=True)
# layer_1 = vgg.features[0].weight.data.numpy() # 64, 3, 3, 3

# For each layer, get the filters
filters = []
for layer in tqdm([28]): #, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28], desc='Getting Layer Filters'): # Conv2D layers
    for i in range(vgg.features[layer].out_channels): # filters
        for j in range(3): # rgb
            filters.append(vgg.features[layer].weight.data.numpy()[i, j, :, :]) # out, in, w, h (or h, w but who cares)
print("Filters got")


# Create permutations of 3x3 matrices
#choices = [-1, 0, .5, 2]
#frames = create_permutations(choices, 9)
frames = []
if method == "randomNumberPermutation":
    frames = create_permutations([0, 1/3, 2/3, 1], 9)
elif method == "randomMatrixGeneration":
    frames = np.random.randint(0, 10, (50000, 9))
else:
    frames = create_permutations([0, 1 / 3, 2 / 3, 1], 9)
print(f"Created {len(frames)} permutations")

# Encode the filters by 'convolving' over each frame
encodings = []
filters = [np.reshape(filter, 9) for filter in filters]
for filter in tqdm(filters, desc='Encoding Filters'):
    encoding = []
    for perm in frames:
       encoding.append(np.dot(filter, perm))
    norm = math.sqrt(np.dot(encoding, encoding))
    encodings.append([x/norm for x in encoding])

# Now compare each encoding by dot producting, keep in mind commutivity
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


if linSort:
    sorted_col_indices = sorted(range(len([row[0] for row in data])), key=lambda k: [row[0] for row in data][k], reverse=True)
    data2 = []
    for row in data:
        data2.append([row[i] for i in sorted_col_indices])
    data3=[data2[i] for i in sorted_col_indices]

if hierSort:
    dist_mat = []
    for row in data:
        dst_row = []
        for i in row:
            dst_row.append(scaled_abs_x(i))
        dist_mat.append(dst_row)

    for i in range(len(dist_mat)):
        assert abs(dist_mat[i][i]) < epsilon
        dist_mat[i][i] = 0
    # https://stats.stackexchange.com/questions/195456/how-to-select-a-clustering-method-how-to-validate-a-cluster-solution-to-warran/195481#195481
    # https://stats.stackexchange.com/questions/195446/choosing-the-right-linkage-method-for-hierarchical-clustering
    methods = ["single", 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']
    for method in methods:
        print("Method:\t", method)

        ordered_dist_mat, res_order, res_linkage = compute_serial_matrix(dist_mat, method)

        plt.pcolormesh(ordered_dist_mat)
        plt.xlim([0, len(dist_mat)])
        plt.ylim([0, len(dist_mat)])
        plt.show(block=False)
        plt.savefig("permutation_mag1_noNeg_thirds_l28/"+str(method)+".png")

if vis:
    figure = plt.figure()
    axes = figure.add_subplot(111)
    caxes = axes.matshow(np.array(data3), interpolation ='none')
    figure.colorbar(caxes)
