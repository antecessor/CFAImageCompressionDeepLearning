from unittest import TestCase
import numpy as np
import cv2
import sys
from compression.Compression import applyLZWCompressionOnImage, decompress
from ImageUtils import compute_psnr


class TestCompress(TestCase):
    def test_compression(self):
        image = cv2.imread('./../data/rgb.jpg')

        image = np.uint8(image * 255)
        compressed, compressionRatio, _, _ = applyLZWCompressionOnImage(image)
        decompressedImage = decompress(compressed, image.shape)
        print("compressed ratio:", compressionRatio)
        psnr = compute_psnr(decompressedImage, image)
        print("PSNR:", psnr)

        TestCase.assertEqual(self, psnr, "Same Image")
