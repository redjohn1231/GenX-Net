import numpy as np
import cv2
import os
import random
from numpy.random import uniform
import threading
import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

tf.random.set_seed(SEED)
cv2.setNumThreads(0)

def gasuss_noise(image, mu=0.0, sigma=0.1):

    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mu, sigma, image.shape)
    gauss_noise = image + noise
    low_clip = 0
    
    gauss_noise = np.clip(gauss_noise, low_clip, 1.0)
    gauss_noise = np.uint8(gauss_noise * 255)
    return gauss_noise

def preprocessing(image,num):
    
    for i in num:
       
        if i==0: 
            image= image
        
        if i==1:
            r = np.random.rand()
            if r<=1:
                jpeg_quality = 50
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
                result, encimg = cv2.imencode('.jpg', image, encode_param)
                decimg = cv2.imdecode(encimg, 1)
                image = decimg
            else:
                image = image
            
        if i==2:
            r = np.random.rand()
            if r<=1:
                sigma =1
                sigma = float(sigma)
                image = cv2.GaussianBlur(image,(3,3),sigma)
            else:
                image = image
                
        if i==3: 
            image = gasuss_noise(image, mu=0.0, sigma=0.1)
            
        if i==4:
            m=random.randint(178,250)
            for i in range(m):
                x=random.randint(0,299)
                y=random.randint(0,299)
                image = cv2.rectangle(image, (x, y), (x+5, y+5), (0, 0, 0), -1)

    return image


def normalization(img,newmin,newmax):
    image = ((img - np.min(img))*(newmax-newmin))/(np.max(img) - np.min(img))+newmin
    return image


def pro(temp,height,width,num):
    
    image=cv2.imread(temp[0])
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image=cv2.resize(image,(height,width,),0,0,cv2.INTER_LINEAR)

    image = preprocessing(image,num)
    
    img = image/255
    
    return img,temp[1]
    

class threadsafe_iter:
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()
    def __iter__(self):
        return self
    def __next__(self): # python3
        with self.lock:
            return self.it.__next__()
def threadsafe_generator(f):
    
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
# num indicates which content-preserving manipulation method is selected.
def generate_data_1(path,label,height=224,width=224,batch_size=40,num=[0]):
    D=[]
    L=[]

    for p in range(len(path)):
        class_path_fake=path[p]
        for fake_name in os.listdir(class_path_fake):
            fake_path=class_path_fake+fake_name
            D.append(fake_path)
            L.append(label[p])
    temp=np.array([D,L]) 
    temp=temp.transpose()
    total_num=temp.shape[0]
  
    while True :
        np.random.shuffle(temp)
        for index in range(0, total_num, (batch_size)):
            x_1=[]
            y_1=[]
            for i in range((batch_size)):
                f=pro(temp[index+i],height,width,num)
                x_1.append(f[0])
                if f[1]=='0':
                    y_1.append([1,0,0,0,0,0,0,0,0])
                if f[1]=='1':
                    y_1.append([0,1,0,0,0,0,0,0,0])
                if f[1]=='2':
                    y_1.append([0,0,1,0,0,0,0,0,0])
                if f[1]=='3':
                    y_1.append([0,0,0,1,0,0,0,0,0])
                if f[1]=='4':
                    y_1.append([0,0,0,0,1,0,0,0,0])
                if f[1]=='5':
                    y_1.append([0,0,0,0,0,1,0,0,0])
                if f[1]=='6':
                    y_1.append([0,0,0,0,0,0,1,0,0])
                if f[1]=='7':
                    y_1.append([0,0,0,0,0,0,0,1,0])
                if f[1]=='8':
                    y_1.append([0,0,0,0,0,0,0,0,1])
            y = np.array(y_1)
            yield (np.array(x_1),(np.array(y)))

           


