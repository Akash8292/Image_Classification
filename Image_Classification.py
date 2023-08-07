import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import tensorflow as tf
from tensorflow.keras import datasets
from keras.optimizers import Adam
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import  accuracy_score
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from tensorflow.keras.preprocessing import image
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Activation, Flatten, Dense, Dropout, Conv2D, MaxPooling2D

# Part 1 - Gathering of large datasets

#Loading cifar10 datasets
(x_train,y_train),(x_test,y_test)=datasets.cifar10.load_data()

# Part 2 - Data Preprocessing

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Normalize to range 0-1
x_test=x_test/255.0
x_train=x_train/255.0
   
# Part 3 - Building the CNN Model 

# Define the model

#Initialize the model
model = Sequential()

# Stacked Convolution layer
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32,32,3),activation = 'relu'))

model.add(Conv2D(32, (3, 3),activation = 'relu'))
 
# Max Pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Dropout for regularization
model.add(Dropout(0.25))

# Stacked Convolution layer
model.add(Conv2D(64, (3, 3), padding='same',activation = 'relu'))

model.add(Conv2D(64, (3, 3),activation = 'relu'))

# Max Pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Dropout for regularization
model.add(Dropout(0.25))

# Stacked Convolution layer
model.add(Conv2D(128, (3, 3), padding='same',activation = 'relu'))

model.add(Conv2D(128, (3, 3),activation = 'relu'))

# Max Pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

# Flattening
model.add(Flatten())

# Fully-connected layer
model.add(Dense(units = 512, activation = 'relu'))

# Dropout for regularization
model.add(Dropout(0.5))

# Fully-connected layer
model.add(Dense(units = 256, activation = 'relu'))

# Dropout for regularization
model.add(Dropout(0.5))

# Output Fully-connected layer
model.add(Dense(10, activation='softmax'))

# Part 4 - Model Training

# Compile the model
model.compile(Adam(lr=0.00015), loss='categorical_crossentropy', metrics=['accuracy'])


# Training the CNN on the Training set
history=model.fit(x_train,y_train,batch_size=64, validation_data=(x_test,y_test),epochs =30)


# Part 5 - Model Evaluation

#Evaluate our CNN Model
model.evaluate(x_test,y_test)

 # Making a prediction
y_pred = model.predict(x_test)

 #Taking maximum predicted outcome vlaue
y_classes = [np.argmax(element) for element in y_pred]

# Accuracy score:
print("Accurancy score:" ,accuracy_score(y_test, y_classes))

# Precision:
print("Precision Score : ",precision_score(y_test,y_classes, 
                                           pos_label='positive'
                                          , average='micro'))
# Recall: 
print("Recall Score : ",recall_score(y_test,y_classes, 
                                           pos_label='positive',
                                           average='macro'))

# F1-Score:
print("F1 score: ",   f1_score(y_test,y_classes, 
                                           pos_label='positive',
                                           average='macro'))

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_classes)
print(cm)

#Summarize history for loss
plt.title('Model Loss')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()

#Summarize history for accuracy
plt.title('Model accuracy')
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()

# Part 6 - Model Optimization

# Data Augmentation
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('static/train_set',
                                                 target_size = (32, 32),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('static/test_set',
                                            target_size = (32, 32),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Part 7 - Input/Ouput 

# Making List of objects
classes=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

# Taking image Input
test_image = image.load_img('static/test_set/bird/0001.png', target_size = (32, 32))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
predict_x=model.predict(test_image) 
classes_x=np.argmax(predict_x,axis=1)

#Predicted output 
def show():
 testImage = img.imread('static/test_set/bird/0001.png') 
 plt.imshow(testImage )

classes_x=int(classes_x)
show()
print("predicted output of given image is :",classes[classes_x])

#model.save('final2_model.h5')






