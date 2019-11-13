# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 17:02:37 2019

@author: prasa
"""

import cv2
from PIL import Image 
im = Image.open("F:/projects/create_training_data/background_images/22-1479814252-ford-figo-rear-seat-01.JPG")
til = Image.new("RGB" , (523,523))
til.paste(im)
til.paste(im,(23,0))
#til.paste(im,(0,23))
til.paste(im,(230,230))
til.save("testtiles.png")