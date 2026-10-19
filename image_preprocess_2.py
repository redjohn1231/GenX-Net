

import numpy as np
import cv2
import os
import random
from numpy.random import uniform
import threading



def gasuss_noise(image, mu=0.0, sigma=0.1):

    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mu, sigma, image.shape)
    gauss_noise = image + noise
    if gauss_noise.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
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
                # sigma = uniform(0,3,size=1)
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



class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()
    def __iter__(self):
        return self
    def __next__(self): # python3
        with self.lock:
            return self.it.__next__()
   
def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator

def generate_data_2(path,label,y2_path,height=224,width=224,batch_size=30,num=[0]):
    D=[]
    L=[]
    
    for p in range(len(path)):
        class_path_fake=path[p]
        for fake_name in os.listdir(class_path_fake):
            fake_path=class_path_fake+fake_name
            D.append(fake_path)
            L.append(label[p])
            
    D = np.array(D)
    L = np.array(L)
    
    y2_all = np.array(y2_path)
    
    total_num = len(D)
    
    indices = np.arange(total_num)
    while True:

       np.random.shuffle(indices)

       for start in range(0, total_num, batch_size):

           batch_idx = indices[start:start + batch_size]

           x_batch = []
           y1_batch = []
           y2_batch = []

           for idx in batch_idx:

               image_path = D[idx]
               y1 = L[idx]
               y2 = y2_all[idx]
               
               image = cv2.imread(image_path)
               image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
               image = cv2.resize(image, (height, width))

               image = preprocessing(image, num)
               image = image / 255.0

               x_batch.append(image)
               y2_batch.append(y2)

               # -------------------------------
               # one-hot
               # -------------------------------
               if y1 == 0:
                   y1_batch.append([1,0,0,0,0,0,0,0,0])
               elif y1 == 1:
                   y1_batch.append([0,1,0,0,0,0,0,0,0])
               elif y1 == 2:
                   y1_batch.append([0,0,1,0,0,0,0,0,0])
               elif y1 == 3:
                   y1_batch.append([0,0,0,1,0,0,0,0,0])
               elif y1 == 4:
                   y1_batch.append([0,0,0,0,1,0,0,0,0])
               elif y1 == 5:
                   y1_batch.append([0,0,0,0,0,1,0,0,0])
               elif y1 == 6:
                   y1_batch.append([0,0,0,0,0,0,1,0,0])
               elif y1 == 7:
                   y1_batch.append([0,0,0,0,0,0,0,1,0])
               elif y1 == 8:
                   y1_batch.append([0,0,0,0,0,0,0,0,1])

           yield (np.array(x_batch),
                 (np.array(y1_batch),
                  np.array(y2_batch, dtype=np.float32)))
  
   