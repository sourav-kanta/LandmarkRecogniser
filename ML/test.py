# -*- coding: utf-8 -*-
"""
Created on Fri May 26 16:02:14 2017

@author: USER PC
"""

import os
import pydot_ng
import numpy
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense
#from keras.layers import Dropout
from keras.layers import Flatten,Input
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras import backend as K
K.set_image_dim_ordering('th')
from sklearn.preprocessing import LabelEncoder
from scipy.misc import imsave,imread,imshow
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD;

path='..\Train_data';

seed = 15
numpy.random.seed(seed)

    
def process_file(path):
    img=imread(path)
    #imshow(img)
    np_img=numpy.asarray(img).astype('float32')/255.0    
    #print np_img.shape
    return np_img
"""
def get_images_in_dir(new_path):
    img_list=[]    
    b_name=[]
    for f in os.listdir(new_path):
        if(f.endswith('.jpg')):
            print os.path.join(new_path,f)
            np_img=process_file(os.path.join(new_path,f))
            #print np_img       
            img_list.append(np_img)
            b_name.append(f)
    return (img_list,b_name)
"""

def get_images():
    img_list=[]    
    b_name=[]
    for fol in os.listdir(path):
        new_path=os.path.join(path,fol)
        for f in os.listdir(new_path):           
            folder=os.path.basename(new_path)
            #print folder 
            #raw_input()
            if(f.endswith('.jpg')):
                #print os.path.join(new_path,f)
                np_img=process_file(os.path.join(new_path,f))
                #print np_img       
                img_list.append(np_img)
                b_name.append(folder)
    return (img_list,b_name)

def do_shuffle(a,b):
    c = numpy.c_[a.reshape(len(a), -1), b.reshape(len(b), -1)]
    numpy.random.shuffle(c)
    a2 = c[:, :a.size//len(a)].reshape(a.shape)
    b2 = c[:, a.size//len(a):].reshape(b.shape)
    return (a2,b2)
    
    

def get_model(length):
    a=Input(shape=(64,64,3))
    c1=Conv2D(64,(3,3),padding='valid', activation='relu',data_format='channels_last')(a)
    m1=MaxPooling2D(pool_size=(2, 2),data_format='channels_last')(c1)
    c2=Conv2D(32,(3,3), padding='valid', activation='relu')(m1)
    m2=MaxPooling2D(pool_size=(2, 2))(c2)
    c3=Conv2D(32,(3,3), padding='valid', activation='relu')(m2)
    m3=MaxPooling2D(pool_size=(2, 2))(c3)
    fl=Flatten()(m3)
    b1=Dense(128, activation='relu')(fl)
    b2=Dense(64, activation='relu')(b1)
    b3=Dense(length,activation='sigmoid')(b2)
    model=Model(inputs=a,outputs=b3)
    plot_model(model,to_file='model.png',show_shapes=True)
    return model


if __name__=="__main__":
    images,names=get_images()    
    encoder = LabelEncoder()
    encoder.fit(names)
    encoded_Y = encoder.transform(names)
    #print names
    Y_cat=np_utils.to_categorical(encoded_Y)
    #Yn=[j for i,j in Y_cat]
    #Yn=numpy.array(Yn).astype('int32')
    #print Yn
    X,Y=(numpy.stack(images),Y_cat)
   
    """
    for i in range(5):
        plt.imshow(Xtrain[i])
        print Ytrain[i]
        raw_input()
        plt.show()
    """
    Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.25,random_state=0)
    print (Ytrain)
    #print "Unique elements : ",len(numpy.unique(Ytrain))
    model=get_model(len(Ytrain[0]))
    
    
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True)
    #model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    #print Xtrain.shape
    batchsize=32
    epochs_no=20
    model.fit(Xtrain,Ytrain,batch_size=batchsize,epochs=epochs_no,verbose=1,validation_split=0.15,shuffle=True)
    result=model.evaluate(Xtest,Ytest,verbose=1)
    print result
    model.save("test_model.h5")
    
