from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import cv2
import tensorflow_datasets as tfds
from keras.datasets import cifar10
import numpy as np
import argparse

from ImageUtils import compute_psnr
from VAENetwork import trainOrGetTrained
from CFAUtils import RGB2CFAUtils
import matplotlib.pyplot as plt

from compression.Compression import applyLZWCompressionOnImage

modelWeightsName = "vae_cnn_compression_CFA.h5"


# def calculateCompressionRatioForAllDecodedImages(images):
#     n, h, w, c = images.shape
#     compressedLatentSpaces = network.predictEncoder(images)
#     compressionRatios = []
#     for i in range(n):
#         decodedImage = network.predictDecoder(np.asarray([compressedLatentSpaces[i]]))
#         error = images[i, :, :, 0] - decodedImage
#         error = np.uint8(error * 255)
#         _, _, compressedSize, uncompressedSize = applyLZWCompressionOnImage(error)
#         compressedSize = sys.getsizeof(network.latent_dim) + compressedSize
#         compressionRatios.append(uncompressedSize / compressedSize)
#     return compressionRatios


# def plotSamples():
#     n = 30
#     figure = np.zeros((image_size * n, image_size * n))
#     # linearly spaced coordinates corresponding to the 2D plot
#     # of digit classes in the latent space
#     grid_x = np.linspace(-4, 4, n)
#     grid_y = np.linspace(-4, 4, n)[::-1]
#     index = 5
#     compressedLatentSpace = network.predictEncoder(x_train)
#     psnr = []
#     for i, yi in enumerate(grid_y):
#         for j, xi in enumerate(grid_x):
#             decodedImage = network.predictDecoder(np.asarray([compressedLatentSpace[index]]))
#             digit = np.round(decodedImage.reshape(image_size, image_size) * 255)
#             psnr.append(compute_psnr(digit, x_train[index]))
#             figure[i * image_size: (i + 1) * image_size,
#             j * image_size: (j + 1) * image_size] = digit
#             index = index + 1
#
#     plt.figure(figsize=(10, 10))
#     start_range = image_size // 2
#     end_range = n * image_size + start_range + 1
#     pixel_range = np.arange(start_range, end_range, image_size)
#     sample_range_x = np.round(grid_x, 1)
#     sample_range_y = np.round(grid_y, 1)
#     plt.xticks(pixel_range, sample_range_x)
#     plt.yticks(pixel_range, sample_range_y)
#     plt.xlabel("z[0]")
#     plt.ylabel("z[1]")
#     plt.imshow(figure, cmap='Greys_r')
#     plt.show()
#     # print(psnr)


def loadFoodDataset():
    food101 = tfds.image.food101.Food101()
    food101.download_and_prepare()
    datasets = food101.as_dataset()
    train_dataset = datasets['train']

    x_test = []
    x_train = []
    for ex in train_dataset:
        image = tfds.as_numpy(ex['image'])
        w, h, c = image.shape
        image = cv2.resize(image, (256, 256))
        if len(x_train) < 1000:
            x_train.append(image)
        else:
            x_test.append(image)
            if len(x_test) >= 1000:
                break
    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)

    return x_train, x_test


def loadCifarData():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    return x_train, x_test


def convertDatasetToCFA(x_train, x_test):
    # Convert to CFA
    rgb2CFAUtils = RGB2CFAUtils()
    n_train, h, w, c = x_train.shape
    n_test, h, w, c = x_test.shape

    x_train_cfa = np.zeros([n_train, h, w, 1])
    for i in range(n_train):
        x_train_cfa[i, :, :, 0] = rgb2CFAUtils.rgb2CFA(x_train[i, :, :, :])[0]
        print("converting image {0} to CFA: Training".format(i))

    x_test_cfa = np.zeros([n_test, h, w, 1])
    for i in range(n_test):
        x_test_cfa[i, :, :, 0] = rgb2CFAUtils.rgb2CFA(x_test[i, :, :, :])[0]
        print("converting image {0} to CFA: Test".format(i))

    image_size = x_train_cfa.shape[1]
    x_train_cfa = np.reshape(x_train_cfa, [-1, image_size, image_size, 1])
    x_test_cfa = np.reshape(x_test_cfa, [-1, image_size, image_size, 1])
    x_train_cfa = x_train_cfa.astype('float32') / 255
    x_test_cfa = x_test_cfa.astype('float32') / 255
    return x_train_cfa, x_test_cfa, image_size


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m", "--mse", help=help_, action='store_true')
    args = parser.parse_args()
    x_train, x_test = loadFoodDataset()
    # x_train, x_test = loadCifarData()
    x_train, x_test, image_size = convertDatasetToCFA(x_train, x_test)
    trainOrGetTrained(x_train, x_test, args.mse, modelWeightsName)
    # plotSamples()
    # compressionRatios = calculateCompressionRatioForAllDecodedImages(x_train)
    # print(compressionRatios)
