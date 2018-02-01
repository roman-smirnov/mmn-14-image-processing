import numpy as np
from scipy import stats as st
from heapq import *

# input image: 4x8 pixels, 8 bits per pixel.
input_img = np.array([21, 21, 21, 95, 169, 243, 243, 243,
                      21, 21, 21, 95, 169, 243, 243, 243,
                      21, 21, 21, 95, 169, 243, 243, 243,
                      21, 21, 21, 95, 169, 243, 243, 243])


def image_entropy(img):
    """ print out input image pixel entropy """

    # pixel count - NxM
    pxl_num = np.alen(img)

    # unique intensity levels - r_k, frequency of each level
    lvls, count = np.unique(img, return_counts=True)

    # compute the histogram - n_k
    im_hist, bin_edges = np.histogram(img, bins=len(lvls))

    # compute the probability of each level - p(r_k)
    probs = count / len(img)
    print('probabilities: ', probs)

    # compute the log_2 of each level probability - lg(p(r_k))
    lvls_prob_log = np.log2(probs)
    print('log_2 of probabilities: ', lvls_prob_log)

    # compute the entropy of the image - \hat{H}
    img_entropy = -np.sum(probs * lvls_prob_log)
    print('image entropy: ', '%.2f' % img_entropy)


def huffman_encode(img):
    """Huffman encode the given image """

    # count the number of unique pairs
    lvls, count = np.unique(img, return_counts=True)

    # build the min heap
    heap = [[heap, [lvls, ""]] for lvls, heap in zip(lvls, count)]
    heapify(heap)

    # extract min leaf and merge until only root left
    while len(heap) > 1:
        lo = heappop(heap)
        hi = heappop(heap)
        # assign code bits
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        # push node with merged count
        heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    # create the codebook dictionary
    codebook = dict(sorted(heappop(heap)[1:], key=lambda p: (len(p[-1]), p)))

    # print the codebook
    print(codebook)
    # print the encoded image
    print(''.join([codebook[v] for v in img]))


def pair_entropy(img):
    """ prints entropy of pixel pair representation """

    # pair up the intensity levels
    pairs = np.reshape(img, (-1, 2))

    print('all pairs: ', pairs)

    # count the number of unique pairs
    unique, count = np.unique(pairs, axis=0, return_counts=True)

    print('unique pairs: ', unique)

    # compute the probability of each pair
    probs = count / len(pairs)

    # calculate the entropy of the pairs, then divide by 2 because each each symbol is 2 pixels
    ent = st.entropy(probs, base=2) / 2

    print('pair entropy: ', ent, 'bits per pixel')


def difference_entropy(img):
    """ prints entropy of pixel difference representation """

    # calculate the pixel differences
    difs = np.diff(img)

    # append the original first element to the start of difference array
    difs = np.append(img[0:1], difs)

    print('pixel differences: ', difs)

    unique, count = np.unique(difs, axis=0, return_counts=True)

    # compute probabilities
    probs = count / len(difs)

    # calculate the pixel difference entropy
    ent = st.entropy(probs, base=2)

    print('pixel difference entropy: ', '%.2f' % ent, 'bits per pixel')


# image_entropy(input_img)
huffman_encode(input_img)
# pair_entropy(input_img)
# difference_entropy(input_img)
