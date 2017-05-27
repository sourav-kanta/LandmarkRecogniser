# -*- coding: utf-8 -*-
"""
Created on Sat May 27 10:40:09 2017

@author: USER PC
"""

from PIL import Image
import os,sys

def resize_image(path,item,dimX=None,dimY=None):
    if not dimX:
        dimX=64
    if not dimY:
        dimY=64
    im=Image.open(os.path.join(path,item))
    imResize = im.resize((dimX,dimY), Image.ANTIALIAS)
    imResize.save(os.path.join(path,item), 'JPEG', quality=90)
    
def resize_dir_img(path,dimX=None,dimY=None):
    if not dimX:
        dimX=64
    if not dimY:
        dimY=64
    for f in os.listdir(path):
        if not f.endswith('.jpg'):
            continue
        try:
            print "Resizing image : ",f
            resize_image(path,f,dimX,dimY)
        except Exception:
            print "Unexpected error:", sys.exc_info()[0]
            pass
    
#resize_image("C:\Users\USER PC\Desktop","test.jpg",200,200)
resize_dir_img("..\Test_Data",64,64)