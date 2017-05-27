# -*- coding: utf-8 -*-
"""
Created on Sat May 27 11:24:06 2017

@author: USER PC
"""

import numpy,os
from scipy.misc import imread
from keras.models import load_model

def process_file(path):
    img=imread(path)
    np_img=numpy.asarray(img).astype('float32')/255.0    
    #print np_img.shape
    return np_img

def get_images_in_dir(new_path):
    img_list=[]    
    b_name=[]
    for f in os.listdir(new_path):
        if(f.endswith('.jpg')):
            #print os.path.join(new_path,f)
            np_img=process_file(os.path.join(new_path,f))
            #print np_img       
            img_list.append(np_img)
            b_name.append(f)
    return (img_list,b_name)


if __name__=='__main__':
    model=load_model('test_model.h5')
    test_img,label=get_images_in_dir("..\Test_Data");
    res=model.predict(numpy.stack(test_img))
    for i in range(len(test_img)):
        print label[i]," has a prediction of : ",res[i]