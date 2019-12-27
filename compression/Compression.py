import sys

import numpy as np
import os
import lz4.frame

compressPath = "../compress/"


def applyLZWCompressionOnImage(image):
    if not os.path.exists(compressPath):
        os.mkdir(compressPath)
    compressedImage = lz4.frame.compress(image)
    originalSize = sys.getsizeof(image)
    compressedSize = sys.getsizeof(compressedImage)
    compressionRatio = originalSize / compressedSize
    return compressedImage, compressionRatio,compressedSize,originalSize


def decompress(compressed, shapes):
    image_bytes = lz4.frame.decompress(compressed)
    return np.reshape(np.frombuffer(image_bytes, np.uint8), shapes)
