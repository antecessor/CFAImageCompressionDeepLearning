from unittest import TestCase

import cv2
from CFAUtils import RGB2CFAUtils


class TestRgb2CFA(TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.rgb2CFAUtils = RGB2CFAUtils()

    def test_convertRGB2CFA(self):
        image = cv2.imread('./../data/rgb.jpg')
        self.rgb2CFAUtils.rgb2CFA(image, True)
