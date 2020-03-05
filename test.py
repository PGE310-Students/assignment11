#!/usr/bin/env python

import unittest
import nbconvert
import os

import skimage
import skimage.measure
import skimage.transform
import cv2
import warnings


with open("assignment11.ipynb") as f:
    exporter = nbconvert.PythonExporter()
    python_file, _ = exporter.from_file(f)

with open("assignment11.py", "w") as f:
    f.write(python_file)

from assignment11 import WellPlot

class TestSolution(unittest.TestCase):
    
    def test_plot(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            p = WellPlot('eog_wells_in_nd.csv')
            p.save_plot()

            gold_image = cv2.imread('images/eog_wells_in_nd_gold.png')
            test_image = cv2.imread('eog_wells_in_nd.png')

            test_image_resized = skimage.transform.resize(test_image, 
                                                          (gold_image.shape[0], gold_image.shape[1]), 
                                                          mode='constant')

            ssim = skimage.measure.compare_ssim(skimage.img_as_float(gold_image), 
                                                test_image_resized, multichannel=True)
            assert ssim >= 0.75

    def test_plot_private(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            p = WellPlot('eog_wells_in_tx.csv')
            p.save_plot()

            gold_image = cv2.imread('images/eog_wells_in_tx_gold.png')
            test_image = cv2.imread('eog_wells_in_tx.png')

            test_image_resized = skimage.transform.resize(test_image, 
                                                          (gold_image.shape[0], gold_image.shape[1]), 
                                                          mode='constant')

            ssim = skimage.measure.compare_ssim(skimage.img_as_float(gold_image), 
                                                test_image_resized, multichannel=True)
            assert ssim >= 0.75


if __name__ == '__main__':
    unittest.main()
