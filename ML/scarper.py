# -*- coding: utf-8 -*-
"""
Created on Sat May 27 19:04:29 2017

@author: USER PC
"""

import os

def name_image(path,start=None):
    if not start:
        start=0
    i=0
    for img in os.listdir(path):
        number=start+i
        new_name=str(number)+".jpg"
        os.rename(os.path.join(path,img),os.path.join(path,new_name))
        print "Renamed : ",img
        i=i+1
        
if __name__=='__main__':
    name_image("C:\Users\USER PC\Desktop\Taj_img\word\media",11)