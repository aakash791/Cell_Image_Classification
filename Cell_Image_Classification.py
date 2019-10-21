#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
print(tf.__version__)


# In[2]:


from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator  
from sklearn.metrics import classification_report, confusion_matrix
#import tensorflow.keras


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


import os
import sys
os.chdir(sys.path[0])


# In[5]:


df=pd.read_csv("gt_training.csv")  
df.head()


# In[6]:


def process_files(x):
    if (len(str(x)) == 1):
        return "0000"+str(x)+".png"
    if (len(str(x)) == 2):
        return "000"+str(x)+".png"
    if (len(str(x)) == 3):
        return "00"+str(x)+".png"
    if (len(str(x)) == 4):
        return "0"+str(x)+".png"
    if (len(str(x)) == 5):
        return str(x)+".png"


# In[7]:


df['Image ID'] = df['Image ID'].map(lambda x: process_files(x))
df.head()


# In[8]:


from PIL import Image
im = Image.open("training/00001.png")
im.size


# In[9]:


from IPython.display import Image 
Image(filename='training/00001.png')


# In[10]:


datagen=ImageDataGenerator(rescale=1./255.,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
train_generator=datagen.flow_from_dataframe(dataframe=df, directory="training", 
                                            x_col="Image ID", y_col="Image class", class_mode="categorical", 
                                            target_size=(78,78), batch_size=113, color_mode='grayscale')


# In[11]:


train_generator.image_shape


# In[12]:


test_datagen=ImageDataGenerator(rescale=1./255.)

test_generator=test_datagen.flow_from_dataframe(dataframe=df,directory="test",
                                                x_col="Image ID", y_col="Image class", class_mode="categorical",
                                                target_size=(78,78), batch_size=113, color_mode='grayscale',shuffle=False)


# In[13]:


test_generator.image_shape


# In[14]:


validation_datagen=ImageDataGenerator(rescale=1./255.)

validation_generator=validation_datagen.flow_from_dataframe(dataframe=df,directory="validation",
                                                x_col="Image ID", y_col="Image class", class_mode="categorical",
                                                target_size=(78,78), batch_size=113, color_mode='grayscale',shuffle=False)


# In[15]:


validation_generator.image_shape


# In[16]:


x_train = train_generator.filenames
y_train = train_generator.labels


# In[17]:


x_val = validation_generator.filenames
y_val = validation_generator.labels


# In[18]:


x_test = test_generator.filenames
y_test = test_generator.labels
label_index = test_generator.class_indices


# In[19]:


model = models.Sequential()
model.add(layers.Conv2D(6, 7, input_shape=(78, 78, 1)))#, activation='relu'))
model.add(layers.LeakyReLU())
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(16, 4))#, activation='relu'))
model.add(layers.LeakyReLU())
model.add(layers.MaxPooling2D((3, 3)))

model.add(layers.Conv2D(32, 3))#, activation='relu'))
model.add(layers.LeakyReLU())
model.add(layers.MaxPooling2D((3, 3)))

model.add(layers.Flatten())
model.add(layers.Dense(150))#, activation='relu'))
model.add(layers.LeakyReLU())
model.add(layers.Dense(6, activation='softmax'))


# In[20]:


model.summary()


# In[21]:


model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01, decay=0.0005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[22]:


cell_model = model.fit_generator(generator=train_generator, validation_data=validation_generator, epochs=1000)


# In[23]:


pred=model.predict_generator(test_generator, verbose=1)  # for confusion matrix


# In[24]:


test_model = model.evaluate_generator(generator=test_generator)
test_model


# In[25]:


y_pred = np.argmax(pred, axis=1)
y_pred.shape


# In[26]:


#confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,6))
plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
plt.title('Confusion matrix', size = 15)
plt.colorbar()
tick_marks = np.arange(6)
plt.xticks(tick_marks, ["0", "1", "2", "3", "4", "5"], rotation=45, size = 6)
plt.yticks(tick_marks, ["0", "1", "2", "3", "4", "5"], size = 6)
plt.tight_layout()
plt.ylabel('Actual label', size = 15)
plt.xlabel('Predicted label', size = 15)
width, height = cm.shape
for x in range(width):
    for y in range(height):
        plt.annotate(str(cm[x][y]), xy=(y, x), horizontalalignment='center', verticalalignment='center')


# In[27]:


print(classification_report(y_test,y_pred))


# In[28]:



#Misclassified images
index = 0
misclassifiedIndexes = []
for label, predict in zip(y_test, y_pred):
    if label != predict: 
        misclassifiedIndexes.append(index)
    index +=1


plt.figure(figsize=(20,4))
for plotIndex, badIndex in enumerate(misclassifiedIndexes[0:5]):
    plt.subplot(1, 5, plotIndex + 1)
    data = image.imread('./test/'+x_test[badIndex])
    plt.imshow(data, cmap=plt.cm.gray)
    plt.title('Predicted: {}, Actual: {}'.format(y_pred[badIndex], y_test[badIndex]), fontsize = 15)


# In[29]:


label_index


# In[30]:


model1 = models.Sequential()
model1.add(layers.Conv2D(6, 7, input_shape=(78, 78, 1)))#, activation='relu'))
model1.add(layers.LeakyReLU())
model1.add(layers.MaxPooling2D((2, 2)))

model1.add(layers.Conv2D(16, 4))#, activation='relu'))
model1.add(layers.LeakyReLU())
model1.add(layers.MaxPooling2D((3, 3)))

model1.add(layers.Conv2D(32, 3))#, activation='relu'))
model1.add(layers.LeakyReLU())
model1.add(layers.MaxPooling2D((3, 3)))

model1.add(layers.Flatten())
model1.add(layers.Dense(150))#, activation='relu'))
model1.add(layers.LeakyReLU())
model1.add(layers.Dense(6, activation='softmax'))


# In[31]:


model1.summary()


# In[32]:


model1.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01, decay=0.0005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[33]:


datagen=ImageDataGenerator(rescale=1./255.,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
train_generator=datagen.flow_from_dataframe(dataframe=df, directory="./training", 
                                            x_col="Image ID", y_col="Image class", class_mode="categorical", 
                                            target_size=(78,78), batch_size=113, color_mode='grayscale')


# In[34]:


test_datagen=ImageDataGenerator(rescale=1./255.)

test_generator=test_datagen.flow_from_dataframe(dataframe=df,directory="./test",
                                                x_col="Image ID", y_col="Image class", class_mode="categorical",
                                                target_size=(78,78), batch_size=113, color_mode='grayscale',shuffle=False)


# In[35]:


cell_model1 = model1.fit_generator(generator=train_generator,validation_data=test_generator, epochs=1000)


# In[36]:


plt.plot(cell_model.history['accuracy'])
plt.plot(cell_model.history['val_accuracy'])
plt.plot(cell_model1.history['val_accuracy'])
plt.legend(['Training','Validation','Testing'])
plt.title('Accuracy')
plt.xlabel('Epochs')


# In[37]:


layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(test_generator)
 
def display_activation(activations, col_size, row_size, act_index): 
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1


# In[38]:


#output after input layer 1
display_activation(activations, 3, 2, 1)


# In[39]:


#output after input layer 4
display_activation(activations, 4, 4, 4)


# In[40]:


#output after input layer 7
display_activation(activations, 8, 4, 7)

