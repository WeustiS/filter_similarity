import torchvision.models as models
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import torch
from util import *
import sys
import os

sys.setrecursionlimit(2000)

def main():

    linSort = False
    hierSort = True
    vis = False
    randMat = False
    generateDistMat = True

    if hierSort:
        assert generateDistMat

    method = "randomMatrixGeneration" if randMat else "numberPermutation"

    # all Conv2D layers [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
    for layer in tqdm([0], desc='Getting Layer Filters'):

        dist_mat = []
        filters = get_filters('vgg', layer)

        frames = get_frames(method)

        encodings = encode_filters(filters, frames)

        dot_relations = compare_encodings(encodings)

        if generateDistMat:
            dist_mat = generate_dist_mat(dot_relations)

        if linSort:
            sorted_col_indices = sorted(range(len([row[0] for row in dot_relations])), key=lambda k: [row[0] for row in dot_relations][k], reverse=True)
            data2 = []
            for row in dot_relations:
                data2.append([row[i] for i in sorted_col_indices])
            dot_relations=[data2[i] for i in sorted_col_indices]


        if hierSort:
            # https://stats.stackexchange.com/questions/195456/how-to-select-a-clustering-method-how-to-validate-a-cluster-solution-to-warran/195481#195481
            # https://stats.stackexchange.com/questions/195446/choosing-the-right-linkage-method-for-hierarchical-clustering
            methods = ["single", 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']
            os.mkdir("default_full_l" + str(layer) + "/")
            for method in methods:
                print("Method:\t", method)

                ordered_dist_mat, res_order, res_linkage = compute_serial_matrix(dist_mat, method)

                figure = plt.figure()
                axes = figure.add_subplot(111)
                caxes = axes.pcolormesh(ordered_dist_mat)
                figure.colorbar(caxes)
                plt.savefig("default_full_l" + str(layer) + "/"+str(method)+".png")
                plt.close(figure)
        if vis:
            figure = plt.figure()
            axes = figure.add_subplot(111)
            caxes = axes.matshow(np.array(dist_mat), interpolation ='none')
            figure.colorbar(caxes)


def get_filters(model_name, layer):
    if model_name == 'vgg':
        vgg = models.vgg16(pretrained=True)
        assert layer in [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28], "You did not select a Conv2D layer for VGG16"
    else:
        raise NameError(f"The model {model_name} is not currently supported by get_filters")
    filters = []
    for i in range(vgg.features[layer].out_channels): # filters
        for j in range(3): # rgb
            filters.append(np.reshape(vgg.features[layer].weight.data.numpy()[i, j, :, :], 9))
            # out, in, w, h (or h, w but who cares)
    filters = np.array(filters)
    return filters


def get_frames(n, method):
    assert n>0, "n must be positive and non-zero"
    frames = []
    if method == "numberPermutation":
        assert n>1, "N cannot be 1 for numberPermutation, there's only 1 permutation. Just like, do [1]*9 bro"
        if n >= 5:
            warnings.warn(
                f"\n----\nFor randomNumberPermutation you are about to generate {n ** 9} items, are you sure?\n----\n")
            ans = input("Y/N")
            if ans.lower() != "y":
                assert False, "Aborted"
        frames = create_permutations(np.arange(0, 1 + epsilon, 1/(n-1)), 9)
    elif method == "randomNumberPermutation":
        assert n > 1, "N cannot be 1 for randomNumberPermutation, there's only 1 permutation. Just like, do [1]*9 bro"
        if n >= 5:
            warnings.warn(f"\n----\nFor randomNumberPermutation you are about to generate {n**9} items, are you sure?\n----\n")
            ans = input("Y/N")
            if ans.lower() != "y":
                assert False, "Aborted"
        frames = create_permutations(np.random.rand(n), 9)
    elif method == "randomMatrixGeneration":
        if n < 10:
            warnings.warn("For randomMatrixGeneration you probably want a very large N")
        frames = np.random.rand(n, 9)
    else:
        assert False, f"{method} is not a method in get_frames"

    return frames


def encode_filters(filters, frames):
    encodings = []
    for filter in tqdm(filters, desc='Encoding Filters'):
        encoding = []
        for perm in frames:
            encoding.append(np.dot(filter, perm))
        norm = math.sqrt(np.dot(encoding, encoding))
        encodings.append([x / norm for x in encoding])

    return encodings


def encode_filters_cuda(filters, frames):
    warnings.warn("This is slower than CPU, for some reason")
    encodings = []

    filters= torch.stack([torch.from_numpy(x).cuda() for x in filters]).cuda()
    frames = torch.stack([torch.from_numpy(x).cuda() for x in frames]).float().cuda()

    for filter in tqdm(filters, desc='Encoding Filters'):
        encoding = []
        for perm in frames:
            encoding.append(torch.dot(filter, perm))
        print("1")
        encoding = torch.Tensor(encoding).cuda()
        print("2")
        norm = torch.sqrt(torch.dot(encoding, encoding))
        encodings.append([x / norm for x in encoding])
        print("3")
    encodings = torch.stack(encodings).cuda()
    return encodings


def compare_encodings(encodings):
    warnings.warn("you probably should use CUDA version")
    data = []
    for i in tqdm(range(len(encodings)), desc='Comparing Encodings'):
        xarr = [0]*len(encodings)
        for j in range(len(encodings)):
            if i<j:
                xarr[j] = np.dot(encodings[i], encodings[j])
            if i == j:
                xarr[j] = 1
            else:
                pass
        data.append(xarr)
    for i in tqdm(range(len(encodings)), desc='Finalizing Encodings'):
        for j in range(len(encodings)):
            if i>j:
                data[i][j] = data[j][i]
    return data

def compare_encodings_cuda(encodings):
    data = []
    if type(encodings) is not type(torch.Tensor()):
        tmp = [torch.Tensor(x).cuda() for x in encodings]
        encodings = torch.stack(tmp).cuda()
        del tmp
    for i in tqdm(range(len(encodings)), desc='Comparing Encodings... with CUDA'):
        xarr = [0]*len(encodings)
        for j in range(len(encodings)):
            if i<j:
                xarr[j] = torch.dot(encodings[i], encodings[j])
            if i==j:
                xarr[j] = 1
            else:
                pass
        data.append(xarr)
    for i in tqdm(range(len(encodings)), desc='Finalizing Encodings'):
        for j in range(len(encodings)):
            if i>j:
                data[i][j] = data[j][i]
    return data

def generate_dist_mat(dot_relations):
    dist_mat = []
    for row in dot_relations:
        dst_row = []
        for i in row:
            dst_row.append(scaled_abs_x(i))
        dist_mat.append(dst_row)

    for i in range(len(dist_mat)):
        assert abs(dist_mat[i][i]) < epsilon
        dist_mat[i][i] = 0
    return dist_mat

if __name__ == "__main__":
    main()
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

'''
Encoding Filters: 100%|██████████| 192/192 [02:45<00:00,  1.16it/s]
Comparing Encodings:  10%|▉         | 19/192 [05:57<54:15, 18.82s/it]
'''