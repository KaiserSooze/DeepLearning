#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
import os
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
import itertools


# In[2]:


import tensorflow
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard


# In[3]:


NUM_CLASSES = 3

CHANNELS = 3

IMAGE_RESIZE = 300

DENSE_LAYER_ACTIVATION = 'softmax'
OBJECTIVE_FUNCTION = 'categorical_crossentropy'

LOSS_METRICS = ['accuracy'] 

NUM_EPOCHS = 30

BATCH_SIZE_TRAINING = 24
BATCH_SIZE_VALIDATION = 16

STEPS_PER_EPOCH_TRAINING = 109 # Training dataset size / Batch size
STEPS_PER_EPOCH_VALIDATION = 55 #Validation dataset size / Batch size

BATCH_SIZE_TESTING = 1


# In[4]:


class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='D:/Vaishnavi/Battery_Test_Samples/logs/001', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tensorflow.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tensorflow.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()


# In[5]:


from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_size = IMAGE_RESIZE

# preprocessing_function is applied on each image but only after re-sizing & augmentation (resize => augment => pre-process)
# Each of the keras.application.resnet* preprocess_input MOSTLY mean BATCH NORMALIZATION (applied on each batch) stabilize the inputs to nonlinear activation functions
# Batch Normalization helps in faster convergence

data_gen = ImageDataGenerator(preprocessing_function = preprocess_input)

# flow_From_directory generates batches of augmented data (where augmentation can be color conversion, etc)
# Both train & valid folders must have NUM_CLASSES sub-folders
train_generator = data_gen.flow_from_directory(
        'D:/Vaishnavi/Battery_Test_Samples/Train',
        target_size=(image_size, image_size),
        batch_size=BATCH_SIZE_TRAINING,
        class_mode='categorical',
        shuffle=True)

validation_generator = data_gen.flow_from_directory(
        'D:/Vaishnavi/Battery_Test_Samples/Validation',
        target_size=(image_size, image_size),
        batch_size=BATCH_SIZE_VALIDATION,
        class_mode='categorical',
        shuffle = True) 


# In[6]:


test_generator = data_gen.flow_from_directory(
    directory = 'D:/Vaishnavi/Battery_Test_Samples/Test',
    target_size = (image_size, image_size),
    batch_size = BATCH_SIZE_TESTING,
    class_mode = None,
    shuffle = False,
    seed = 123
)


# In[ ]:


vgg19_model = tensorflow.keras.applications.vgg19.VGG19(weights=None, input_shape=(300, 300, 3))


# In[ ]:


model = Sequential()
for layer in vgg19_model.layers[:-1]:
    model.add(layer)
for layer in model.layers:
    layer.trainable = False
model.add(Dense(2, activation='softmax'))


# In[ ]:





# In[7]:


from tensorflow.keras import optimizers

adam = optimizers.Adam(lr = 0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)


# In[ ]:


model.compile(optimizer = adam, loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


start_time = datetime.now()
print('The program started on: {}'.format(start_time))
early_stopping = EarlyStopping(patience=10)

fit_history = model.fit_generator(train_generator, steps_per_epoch=109, callbacks= [early_stopping, TrainValTensorBoard(write_graph=True)], validation_data=validation_generator, validation_steps=55, epochs=50, verbose=2)
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# In[ ]:


model.save('vgg19_with_color.h5')


# In[ ]:


plt.figure(1, figsize = (15,8)) 
    
plt.subplot(221)  
plt.plot(fit_history.history['acc'])  
plt.plot(fit_history.history['val_acc'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'valid']) 
    
plt.subplot(222)  
plt.plot(fit_history.history['loss'])  
plt.plot(fit_history.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'valid']) 

plt.show()


# In[ ]:




pred = model.predict_generator(test_generator, steps = len(test_generator), verbose = 1)

predicted_class_indices = np.argmax(pred, axis = 1)


# In[ ]:




print('Accuracy Score:', accuracy_score(test_generator.classes, predicted_class_indices))


# In[ ]:


cm = confusion_matrix(test_generator.classes, predicted_class_indices)


# In[ ]:


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fig = plt.gcf()
    fig.set_size_inches (7, 7)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized Confusion Matrix")
    else:
        print('Confusion Matrix, without normalization')
        
    print(cm)
    
    thresh = cm.max() / 2.
    for i, j  in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="black" if cm[i, j] > thresh else "black", fontsize = 14, weight = 'bold')
        
    plt.tight_layout()
    plt.rcParams.update({'font.size': 30})
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')


# In[ ]:


cm_plot_labels = ['Good','Bad']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')


# In[8]:


new_model = tensorflow.keras.applications.vgg19.VGG19(input_shape=(300, 300, 3), weights=None)
#weights_model =r'D:\Vaishnavi\Battery_Test_Samples\vgg19_with_color.h5'
model1 = Sequential()
for layer in new_model.layers[:-1]:
    model1.add(layer)
for layer in model1.layers[:8]:
    layer.trainable = False

model1.add(Dense(2, activation='softmax'))
#model1.load_weights(weights_model)
    


# In[9]:


for i,layer in enumerate(model1.layers):
    print(i,layer.name,layer.trainable)


# In[10]:


model1.compile(optimizer = adam, loss='categorical_crossentropy', metrics=['accuracy'])


# In[11]:


start_time = datetime.now()
print('The program started on: {}'.format(start_time))

fit_history = model1.fit_generator(train_generator, steps_per_epoch=109, validation_data=validation_generator, validation_steps=55, epochs=5, verbose=2)
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# In[12]:


plt.figure(1, figsize = (15,8)) 
    
plt.subplot(221)  
plt.plot(fit_history.history['acc'])  
plt.plot(fit_history.history['val_acc'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'valid']) 
    
plt.subplot(222)  
plt.plot(fit_history.history['loss'])  
plt.plot(fit_history.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'valid']) 

plt.show()


# In[13]:


pred = model1.predict_generator(test_generator, steps = len(test_generator), verbose = 1)

predicted_class_indices = np.argmax(pred, axis = 1)


# In[14]:


print('Accuracy Score:', accuracy_score(test_generator.classes, predicted_class_indices))


# In[15]:


cm = confusion_matrix(test_generator.classes, predicted_class_indices)


# In[18]:


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fig = plt.gcf()
    fig.set_size_inches (7, 7)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized Confusion Matrix")
    else:
        print('Confusion Matrix, without normalization')
        
    print(cm)
    
    thresh = cm.max() / 2.
    for i, j  in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="black" if cm[i, j] > thresh else "black", fontsize = 14, weight = 'bold')
        
    plt.tight_layout()
    plt.rcParams.update({'font.size': 30})
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')


# In[20]:


cm_plot_labels = ['Good','Bad']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')


# In[ ]:




