import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
import os 

path = 'C:\\Users\\user\\Desktop\\CNN-real-dataset\\images'


TEST_DIR = path +'\\TEST'
TRAIN_DIR = path + '\\TRAIN'

## imshow an image

dir =  os.listdir(TRAIN_DIR + '\\MONOCYTE')
sample_image = plt.imread(TRAIN_DIR +'\\MONOCYTE\\'+ dir[0]) 

#plt.figure(figsize=(10,10))
#plt.imshow(sample_image)


dirs = os.listdir(TRAIN_DIR)

X_train = []
X_test  = []
Y_train = []
Y_test  = []


for sub_dirs in dirs:
    for image_path in os.listdir(TRAIN_DIR+'\\'+sub_dirs):
        image = plt.imread(TRAIN_DIR+'\\'+sub_dirs +'\\'+ image_path)
        Y_train.append(dirs.index(sub_dirs))
        X_train.append(image/255)

    for image_path in os.listdir(TEST_DIR+'\\'+sub_dirs):
        image = plt.imread(TEST_DIR+'\\'+sub_dirs +'\\'+ image_path)
        Y_test.append(dirs.index(sub_dirs))
        X_test.append(image/255)


X_train = np.array(X_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)
Y_train = np.array(Y_train)
from tensorflow.keras.utils import to_categorical

Y_train_cat = to_categorical(Y_train)
Y_test_cat  = to_categorical(Y_test)



### MODELL


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPool2D
model = Sequential()

model.add(Conv2D(filters=16,kernel_size=(4,4),strides=(1,1),activation='relu',input_shape=(240,320,3)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=32,kernel_size=(4,4),strides=(1,1),activation='relu',input_shape=(240,320,3)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(4,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

model.summary()

from tensorflow.keras.callbacks import EarlyStopping
early = EarlyStopping(monitor='val_loss',patience=1)

model.fit(X_train,Y_train_cat,validation_data=(X_test,Y_test_cat),callbacks=[early],batch_size=64,epochs=5,verbose=1)

### evalutaion

preds = model.predict_classes(X_test)

from sklearn.metrics import confusion_matrix,classification_report

print(confusion_matrix(Y_test,preds))
print(classification_report(Y_test,preds))




