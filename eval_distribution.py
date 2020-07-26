from filter_sim import get_filters, get_frames, encode_filters, compare_encodings, generate_dist_mat, vgg


def main(vgg):
    dist_mat = generate_dist_mat(compare_encodings(encode_filters(get_filters('vgg', 0), get_frames('randomNumberPermutation'))))
    # that was a fun line to write

def eval_dist_mat(dist_mat):

if __name__ == "__main__":
    main(vgg)