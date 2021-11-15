import skimage
import skimage.io
import skimage.transform
import os
import numpy as np
import utils
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # DO NOT CHANGE
    impath = os.path.join("images", "noisy_moon.png")
    im = utils.read_im(impath)

    # START YOUR CODE HERE ### (You can change anything inside this block)
    # These two variables create the indices for filtering the image correctly, leaving a hole in the middle
    y_indicies = np.concatenate((np.arange(0, int(im.shape[1] / 2 - 12)), np.arange(int(im.shape[1] / 2 + 12), im.shape[1] - 1)));
    x_indicies = np.arange(int(im.shape[0] / 2 - 1), int(im.shape[0] / 2 + 3))

    F = np.fft.fft2(im)
    kernel = np.ones_like((im))
    for i in x_indicies:
        kernel[i, y_indicies] = 0
    kernel = np.fft.fftshift(kernel)
    H = kernel
    G = F * H
    im_filtered = np.fft.ifft2(G).real
    plt.imshow(np.log(abs(np.fft.fftshift(F)) + 1), cmap="gray")
    plt.show()


    plt.figure(figsize=(20, 4))
    # plt.subplot(num_rows, num_cols, position (1-indexed))
    plt.subplot(1, 5, 1)
    plt.imshow(im, cmap="gray")
    plt.subplot(1, 5, 2)
    # Visualize FFT
    plt.imshow(np.log(abs(np.fft.fftshift(F)) + 1), cmap="gray")
    plt.subplot(1, 5, 3)
    # Visualize FFT kernel
    plt.imshow(np.log(abs(np.fft.fftshift(H)) + 1), cmap="gray")
    plt.subplot(1, 5, 4)
    # Visualize filtered FFT image
    plt.imshow(np.log(abs(np.fft.fftshift(G)) + 1), cmap="gray")
    plt.subplot(1, 5, 5)
    # Visualize filtered spatial image
    plt.imshow(im_filtered, cmap="gray")
    plt.show()

    ### END YOUR CODE HERE ###
    utils.save_im("moon_filtered.png", utils.normalize(im_filtered))
