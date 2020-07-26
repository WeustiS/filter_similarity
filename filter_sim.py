import torchvision.models as models
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import torch
from util import *
import sys
import os

sys.setrecursionlimit(2000)
vgg = models.vgg16(pretrained=True)

def main():

    linSort = False
    hierSort = True
    vis = False
    randMat = False
    generateDistMat = True

    if hierSort:
        assert generateDistMat

    method = "randomMatrixGeneration" if randMat else "randomNumberPermutation"



    for layer in tqdm([0], desc='Getting Layer Filters'): # all Conv2D layers [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
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
            filters.append(vgg.features[layer].weight.data.numpy()[i, j, :, :]) # out, in, w, h (or h, w but who cares)
    return filters

def get_frames(method):
    frames = []
    if method == "randomNumberPermutation":
        frames = create_permutations([0, .5, 1], 9)
    elif method == "randomMatrixGeneration":
        frames = np.random.randint(0, 10, (50000, 9))
    else:
        frames = create_permutations([0, 1 / 3, 2 / 3, 1], 9)
    return frames

def encode_filters(filters, frames):
    encodings = []
    filters = [np.reshape(filter, 9) for filter in filters]
    for filter in tqdm(filters, desc='Encoding Filters'):
        encoding = []
        for perm in frames:
            encoding.append(np.dot(filter, perm))
        norm = math.sqrt(np.dot(encoding, encoding))
        encodings.append([x / norm for x in encoding])
    return encodings

def compare_encodings(encodings):
    data = []
    for i in tqdm(range(len(encodings)), desc='Comparing Encodings'):
        xarr = [0]*len(encodings)
        for j in range(len(encodings)):
            if i<=j:
                xarr[j] = np.dot(encodings[i], encodings[j])
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