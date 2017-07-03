# -*- coding: utf-8 -*-
"""
Created on Sat May 27 11:24:06 2017

@author: USER PC
"""

import numpy,os
from scipy.misc import imread
from keras.models import model_from_json

names=['Taj Mahal','Vidhan Soudha']

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
    json_file=open("model.json","r")
    loaded_json=json_file.read()
    json_file.close()
    
    model=model_from_json(loaded_json)
    model.load_weights("model.h5")
    test_img,label=get_images_in_dir("..\Test_Data");
    res=model.predict(numpy.stack(test_img))
    print res
    numpy.set_printoptions(suppress=True)
    for i in range(len(test_img)):
        
        """
        if res[i][1]<=res[i][2]:
            output="Vidhan Soudha"
        else:
            output="Taj Mahal"
        """
        #resb=res[i]
        resb=res[i].tolist()
        ind=resb.index(max(resb))
        output=names[ind]+""
        print label[i]," has a prediction of : ",output," with ",res[i]