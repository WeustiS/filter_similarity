from filter_sim import *


def main():
    for i in [0, 2, 5, 7, 10]:
        dist_mat_2 = generate_dist_mat(
                    compare_encodings_cuda(
                        encode_filters(
                            get_filters('vgg', i),
                            get_frames(2, 'randomNumberPermutation')
                        )))
        dist_mat_3 = generate_dist_mat(
                    compare_encodings_cuda(
                        encode_filters(
                            get_filters('vgg', i),
                            get_frames(3, 'randomNumberPermutation')
                        )))
        dist_mat_4 = generate_dist_mat(
                    compare_encodings_cuda(
                        encode_filters(
                            get_filters('vgg', i),
                            get_frames(4, 'randomNumberPermutation')
                        )))
        # that was a fun line to write
        score_2, stddev_2 = eval_dist_mat(dist_mat_2)
        score_3, stddev_3 = eval_dist_mat(dist_mat_3)
        score_4, stddev_4 = eval_dist_mat(dist_mat_4)

        l1_error_23, l2_error_23 = compare_dist_mat(dist_mat_2, dist_mat_3)
        l1_error_34, l2_error_34 = compare_dist_mat(dist_mat_3, dist_mat_4)
        l1_error_24, l2_error_24 = compare_dist_mat(dist_mat_2, dist_mat_4)
        print(f"LAYER {i}")
        print(score_2, stddev_2)
        print(score_3, stddev_3)
        print(score_4, stddev_4)
        print(l1_error_23, l2_error_23)
        print(l1_error_34, l2_error_34)
        print(l1_error_24, l2_error_24)

'''
number permutations, 2, 3, 4
LAYER 0
tensor(0.6097, device='cuda:0') tensor(0.3147, device='cuda:0')
tensor(0.6016, device='cuda:0') tensor(0.3296, device='cuda:0')
tensor(0.5999, device='cuda:0') tensor(0.3325, device='cuda:0')
tensor(565.9182, device='cuda:0') tensor(6.1710, device='cuda:0')
tensor(111.4100, device='cuda:0') tensor(1.2116, device='cuda:0')
tensor(673.4180, device='cuda:0') tensor(7.3454, device='cuda:0')

LAYER 2
tensor(0.5253, device='cuda:0') tensor(0.2913, device='cuda:0')
tensor(0.3032, device='cuda:0') tensor(0.2899, device='cuda:0')
tensor(0.4668, device='cuda:0') tensor(0.3017, device='cuda:0')
tensor(4468.6479, device='cuda:0') tensor(38.7206, device='cuda:0')
tensor(3277.7424, device='cuda:0') tensor(28.7491, device='cuda:0')
tensor(1352.5439, device='cuda:0') tensor(11.8438, device='cuda:0')

LAYER 5
tensor(0.5026, device='cuda:0') tensor(0.2794, device='cuda:0')
tensor(0.4528, device='cuda:0') tensor(0.2871, device='cuda:0')
tensor(0.4397, device='cuda:0') tensor(0.2883, device='cuda:0')
tensor(4618.8853, device='cuda:0') tensor(19.9665, device='cuda:0')
tensor(1184.1735, device='cuda:0') tensor(5.1083, device='cuda:0')
tensor(5752.3047, device='cuda:0') tensor(24.8552, device='cuda:0')

LAYER 7
tensor(0.4799, device='cuda:0') tensor(0.2798, device='cuda:0')
tensor(0.4012, device='cuda:0') tensor(0.2851, device='cuda:0')
tensor(0.4118, device='cuda:0') tensor(0.2850, device='cuda:0')
tensor(6679.6455, device='cuda:0') tensor(28.5184, device='cuda:0')
tensor(890.4374, device='cuda:0') tensor(3.8101, device='cuda:0')
tensor(5824.1519, device='cuda:0') tensor(24.8666, device='cuda:0')

LAYER 10
tensor(0.4227, device='cuda:0') tensor(0.2850, device='cuda:0')
tensor(0.4183, device='cuda:0') tensor(0.2852, device='cuda:0')
tensor(0.3568, device='cuda:0') tensor(0.2855, device='cuda:0')
tensor(1523.7008, device='cuda:0') tensor(3.2613, device='cuda:0')
tensor(20552.5039, device='cuda:0') tensor(44.2601, device='cuda:0')
tensor(22026.6133, device='cuda:0') tensor(47.4062, device='cuda:0')
'''

'''
random number permutations, 2, 3, 4 (running)

LAYER 0
tensor(0.4710, device='cuda:0') tensor(0.3579, device='cuda:0')
tensor(0.5923, device='cuda:0') tensor(0.3437, device='cuda:0')
tensor(0.5922, device='cuda:0') tensor(0.3439, device='cuda:0')
tensor(3330.1567, device='cuda:0') tensor(33.4956, device='cuda:0')
tensor(7.6849, device='cuda:0') tensor(0.0833, device='cuda:0')
tensor(3325.0862, device='cuda:0') tensor(33.4475, device='cuda:0')

LAYER 2
tensor(0.3186, device='cuda:0') tensor(0.2933, device='cuda:0')
tensor(0.3124, device='cuda:0') tensor(0.2920, device='cuda:0')
tensor(0.3682, device='cuda:0') tensor(0.3013, device='cuda:0')
tensor(122.0569, device='cuda:0') tensor(1.1106, device='cuda:0')
tensor(1115.2152, device='cuda:0') tensor(10.0155, device='cuda:0')
tensor(995.6165, device='cuda:0') tensor(8.9311, device='cuda:0')

LAYER 5
tensor(0.1836, device='cuda:0') tensor(0.2436, device='cuda:0')
tensor(0.3530, device='cuda:0') tensor(0.2878, device='cuda:0')
tensor(0.3345, device='cuda:0') tensor(0.2859, device='cuda:0')
tensor(13097.7832, device='cuda:0') tensor(59.5313, device='cuda:0')
tensor(1536.7844, device='cuda:0') tensor(6.7658, device='cuda:0')
tensor(11661.5381, device='cuda:0') tensor(53.4924, device='cuda:0')

LAYER 7
tensor(0.2013, device='cuda:0') tensor(0.2501, device='cuda:0')
tensor(0.3034, device='cuda:0') tensor(0.2773, device='cuda:0')
tensor(0.2562, device='cuda:0') tensor(0.2673, device='cuda:0')
tensor(7880.4165, device='cuda:0') tensor(36.0252, device='cuda:0')
tensor(3702.4109, device='cuda:0') tensor(16.6058, device='cuda:0')
tensor(4234.2725, device='cuda:0') tensor(19.8365, device='cuda:0')
'''

'''
n^^9 random matrices (console)
LAYER 0
tensor(0.5503, device='cuda:0') tensor(0.3698, device='cuda:0')
tensor(0.5832, device='cuda:0') tensor(0.3544, device='cuda:0')
tensor(0.5769, device='cuda:0') tensor(0.3599, device='cuda:0')
tensor(1190.1285, device='cuda:0') tensor(12.6842, device='cuda:0')
tensor(289.3646, device='cuda:0') tensor(3.1148, device='cuda:0')
tensor(928.1562, device='cuda:0') tensor(9.8708, device='cuda:0')

LAYER 2
tensor(0.1776, device='cuda:0') tensor(0.2435, device='cuda:0')
tensor(0.5177, device='cuda:0') tensor(0.2931, device='cuda:0')
tensor(0.4161, device='cuda:0') tensor(0.3043, device='cuda:0')
tensor(6495.0225, device='cuda:0') tensor(55.9000, device='cuda:0')
tensor(2222.9636, device='cuda:0') tensor(19.4022, device='cuda:0')
tensor(4545.6309, device='cuda:0') tensor(40.9399, device='cuda:0')


'''

def eval_dist_mat(dist_mat):
    total_distance = 0
    total_items = 0

    for i in range(len(dist_mat)):
        for j in range(i): # fine bc symmetrical mat
            total_distance += dist_mat[i][j]
            total_items += 1
    avg = total_distance/total_items
    total_error = 0

    for i in range(len(dist_mat)):
        for j in range(i): # fine bc symmetrical mat
            total_error+=(dist_mat[i][j] - avg)**2
    total_error = (total_error/total_items)**.5

    return avg, total_error

def compare_dist_mat(dist_mat_0, dist_mat_1):
    # we do L2
    assert len(dist_mat_0) == len(dist_mat_1), "Distance matrices are of different size"
    assert len(dist_mat_0[0]) == len(dist_mat_1[0]), "Distance matrices are of different size"
    l1_error = 0 # more resistant to outliers
    l2_error = 0

    for i in range(len(dist_mat_0)):
        for j in range(i):  # fine bc symmetrical
            l1_error += abs(dist_mat_0[i][j] - dist_mat_1[i][j])
            l2_error += (dist_mat_0[i][j] - dist_mat_1[i][j])**2
    l2_error = l2_error**.5

    return l1_error, l2_error
if __name__ == "__main__":
    main()