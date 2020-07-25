import torchvision.models as models
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import torch
from util import *
import sys
import os

sys.setrecursionlimit(2000)

linSort = False
hierSort = True
vis = False
randMat = False
method = "randomMatrixGeneration" if randMat else "randomNumberPermutation"

vgg = models.vgg16(pretrained=True)
# layer_1 = vgg.features[0].weight.data.numpy() # 64, 3, 3, 3

# For each layer, get the filters

for layer in tqdm([0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28], desc='Getting Layer Filters'): # Conv2D layers
    filters = []
    for i in range(vgg.features[layer].out_channels): # filters
        for j in range(3): # rgb
            filters.append(vgg.features[layer].weight.data.numpy()[i, j, :, :]) # out, in, w, h (or h, w but who cares)
    print("Filters got")
    # Create permutations of 3x3 matrices
    #choices = [-1, 0, .5, 2]
    #frames = create_permutations(choices, 9)
    frames = []
    if method == "randomNumberPermutation":
        frames = create_permutations([0, .5, 1], 9)
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
        os.mkdir("default_full_l" + str(layer) + "/")
        for method in methods:
            print("Method:\t", method)

            ordered_dist_mat, res_order, res_linkage = compute_serial_matrix(dist_mat, method)

            plt.pcolormesh(ordered_dist_mat)
            plt.xlim([0, len(dist_mat)])
            plt.ylim([0, len(dist_mat)])
            # plt.show(block=False)
            plt.colorbar()
            plt.savefig("default_full_l" + str(layer) + "/"+str(method)+".png")

    if vis:
        figure = plt.figure()
        axes = figure.add_subplot(111)
        caxes = axes.matshow(np.array(dist_mat), interpolation ='none')
        figure.colorbar(caxes)
# benefits of filter diversity, measurements https://arxiv.org/pdf/2004.03334v2.pdf
# benefits of filter diversity https://github.com/ddhh/NoteBook/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/LNCS%207700%20Neural%20Networks%20Tricks%20of%20the%20Trade.pdf
# weight variance on quality http://cce.lternet.edu/docs/bibliography/Public/377ccelter.pdf
# " In theory, the larger the filter variance, the more important the filter is. https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8786196

'''
https://arxiv.org/pdf/1906.04252.pdf
https://openaccess.thecvf.com/content_ICCV_2017/papers/Zoumpourlis_Non-Linear_Convolution_Filters_ICCV_2017_paper.pdf
https://www.machinecurve.com/index.php/2018/12/07/convolutional-neural-networks-and-their-components-for-computer-vision/
https://misl.ece.drexel.edu/wp-content/uploads/2018/04/BayarStammTIFS01.pdf
https://medium.com/@bairoukanasa5/improving-convolutional-neural-network-accuracy-using-gabor-filter-and-progressive-resizing-8e60caf50d8d
https://arxiv.org/ftp/arxiv/papers/1904/1904.13204.pdf
https://www.researchgate.net/post/Fixed_Gabor_Filter_in_Convolutional_Neural_Networks
https://elib.dlr.de/117228/1/Masterthesis.pdf
https://towardsdatascience.com/demystifying-convolutional-neural-networks-384785791596
https://medium.com/@eos_da/applying-neural-network-and-local-laplace-filter-methods-to-very-high-resolution-satellite-imagery-3b203e5cc444
http://papers.nips.cc/paper/4061-layer-wise-analysis-of-deep-networks-with-gaussian-kernels.pdf
https://arxiv.org/pdf/1803.00388.pdf
https://vcl.iti.gr/new/volterra-based-convolution-filter-implementation-in-torch/
https://www.google.com/search?q=filter+normalization+convolutional+neural+network&oq=filter+normalization+conv&aqs=chrome.1.69i57j33l4.5731j0j7&sourceid=chrome&ie=UTF-8
https://stats.stackexchange.com/questions/133368/how-to-normalize-filters-in-convolutional-neural-networks
https://arxiv.org/pdf/1911.09737.pdf
'''