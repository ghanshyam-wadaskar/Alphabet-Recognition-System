#!/usr/bin/env python
# coding: utf-8

# ### 1. import data

# In[1]:


from mnist import MNIST


# In[2]:


data = MNIST(path='E:\Anaconda\Data', return_type='numpy')
data.select_emnist('letters')
X, y = data.load_training()


# In[3]:


X = X.reshape(124800, 28, 28)
y = y.reshape(124800, 1)


# In[4]:


# list(y) --> y ranges from 1 to 26


# In[5]:


y = y-1


# In[6]:


#list(y) --> y ranges from 0 to 25 now


# ### 2. train-test split
# 

# In[7]:


# pip install scikit-learn


# In[8]:


# pip install scikit-learn
from sklearn.model_selection import train_test_split


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=50)


# In[10]:


#(0,255) --> (0,1)
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255


# In[11]:


#y_train, y_test


# In[12]:


# pip install tensorflow


# In[13]:


# pip install tensorflow
# integer into one hot vector (binary class matrix)
from keras.utils import to_categorical

# Assuming y_train and y_test are your class labels as integers
y_train = to_categorical(y_train, num_classes=26)
y_test = to_categorical(y_test, num_classes=26)


# In[14]:


#y_train, y_test


# ### 3. Define our model

# In[15]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
 
model = Sequential()
model.add(Flatten(input_shape = (28,28)))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2)) # preventing overfitting
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(26, activation='softmax'))


# In[16]:


model.summary()


# In[17]:


model.compile(loss= 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])


# ### 4. calculate accuracy
# 

# In[18]:


# before training
score = model.evaluate(X_test, y_test, verbose=0)
accuracy = 100*score[1]
print("Before training, test accuracy is", accuracy)


# In[19]:


# let's train our model
from keras.callbacks import ModelCheckpoint
 
checkpointer = ModelCheckpoint(filepath = 'best_model.h5', verbose=1, save_best_only = True)
model.fit(X_train, y_train, batch_size = 128, epochs= 10, validation_split = 0.2, 
          callbacks=[checkpointer], verbose=1, shuffle=True)


# In[20]:


model.load_weights('best_model.h5')


# In[21]:


# calculate test accuracy
score = model.evaluate(X_test, y_test, verbose=0)
accuracy = 100*score[1]
 
print("Test accuracy is ", accuracy)


# ### part 2: Alphabet Recognition System

# In[22]:


from keras.models import load_model


# In[23]:


model = load_model('best_model.h5')


# In[24]:


letters ={ 0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h', 8:'i', 9:'j', 10:'k', 11:'l', 
          12:'m', 13:'n', 14:'o', 15:'p', 16:'q', 17:'r', 18:'s', 19:'t', 20:'u', 21:'v', 22:'w', 
          23:'x', 24:'y', 25:'z', 26:''}


# In[25]:


# pip install numpy


# In[26]:


# defining blue color in hsv format
# pip install numpy
import numpy as np
 
blueLower = np.array([100,60,60])
blueUpper = np.array([140,255,255])


# In[27]:


kernel = np.ones((5,5), np.uint8)


# In[28]:


# define blackboard
blackboard = np.zeros((480,640, 3), dtype=np.uint8)
alphabet = np.zeros((200,200,3), dtype=np.uint8)


# In[29]:


# deques (Double ended queue) is used to store alphabet drawn on screen
from collections import deque
points = deque(maxlen = 512)


# In[30]:


# pip install opencv-python


# ### open the camera and recognize alphabet

# In[31]:


import cv2 #pip install opencv-python
cap = cv2.VideoCapture(0)
prediction = 26
while True:
    ret, frame=cap.read()
    frame = cv2.flip(frame, 1)
     
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     
    # Detecting which pixel value falls under blue color boundaries
    blue = cv2.inRange(hsv, blueLower, blueUpper)
     
    #erosion
    blue = cv2.erode(blue, kernel)
    #opening
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, kernel)
    #dilation
    blue = cv2.dilate(blue, kernel)
     
    # find countours in the image
    cnts , _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      
    center = None
     
    # if any countours were found
    if len(cnts) > 0:
        cnt = sorted(cnts, key = cv2.contourArea, reverse=True)[0]
        ((x,y), radius) = cv2.minEnclosingCircle(cnt)
        cv2.circle(frame, (int(x), int(y),), int(radius), (125,344,278), 2)
         
        M = cv2.moments(cnt)
        center = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
     
        points.appendleft(center)
         
    elif len(cnts) == 0:
        if len(points) != 0:
            blackboard_gray = cv2.cvtColor(blackboard, cv2.COLOR_BGR2GRAY)
            blur = cv2.medianBlur(blackboard_gray, 15)
            blur = cv2.GaussianBlur(blur, (5,5), 0)
            thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            #cv2.imshow("Thresh", thresh)
             
            blackboard_cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
             
            if len(blackboard_cnts)>=1:
                cnt = sorted(blackboard_cnts, key=cv2.contourArea, reverse=True)[0]
                 
                if cv2.contourArea(cnt)>1000:
                    x,y,w,h = cv2.boundingRect(cnt)
                    alphabet = blackboard_gray[y-10:y+h+10,x-10:x+w+10]
                    try:
                        img = cv2.resize(alphabet, (28,28))
                    except cv2.error as e:
                        continue
                     
                    img = np.array(img)
                    img = img.astype('float32')/255
                     
                    prediction = model.predict(img.reshape(1,28,28))[0]
                    prediction = np.argmax(prediction)
                     
            # Empty the point deque and also blackboard
            points = deque(maxlen=512)
            blackboard = np.zeros((480,640, 3), dtype=np.uint8)
         
    # connect the detected points with line
    for i in range(1, len(points)):
        if points[i-1] is None or points[i] is None:
            continue
        cv2.line(frame, points[i-1], points[i], (0,0,0), 2)
        cv2.line(blackboard, points[i-1], points[i], (255,255,255), 8)
         
     
    cv2.putText(frame, "Prediction: " + str(letters[int(prediction)]), (20,400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
     
     
    cv2.imshow("Alphabet Recognition System", frame)
     
    if cv2.waitKey(1)==13: #if I press enter
        break
cap.release()
cv2.destroyAllWindows()


# In[ ]:




