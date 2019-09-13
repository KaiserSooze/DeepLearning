#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

from datetime import datetime
from sklearn.utils import resample
np.random.seed(7)


# In[2]:


root_train = r'C:\Users\sanatara\DeepLearning\Scripts\WeldingInspection\TimeSeries\data\20190725_Precitec\AR3000_Stihl\Defocus\Train'
#root_test = r'C:\Users\sanatara\DeepLearning\Scripts\WeldingInspection\TimeSeries\data\20190725_Precitec\AR3000_Stihl\Defocus\Test'


# In[3]:

fileorigins =[]

# In[4]:
#Function for prepreocessing your data 
def preprocess_train_test(root):
    allfiles =[]
    split=[]
    fileorigins.clear()
    for root, dirs, files in os.walk(root):
        data = [os.path.join(root,f) for f in files if f.endswith(".txt")]
        for file in data:
            dest = file.split("\\")  
            classification = dest[-2]
            name = dest[-1]
            s = "\\\\"
            dest= s.join(dest)
            frame = pd.read_csv(str(dest), delim_whitespace=True,header=0, index_col=0,skiprows=9)
            frame['classification']= classification
            frame=frame.loc[:, ~frame.columns.str.replace("(\.\d+)$", "").duplicated()]
            df = pd.DataFrame(data = frame)
            if "Zeit" or "Fehler" in df:
                df = df.rename(columns={'Zeit':'Time', 'Fehler':'Error'})

            a = df.drop(['Time','P-Ref+','Plasma','Error','P-Ref-','P-MW','T-Ref+','T-Ref-','T-MW','T-Raw','R-Ref+','Refl','R-Ref-','R-MW'],axis=1)
                     
            df1 = a[0:753]
            df2 = a[753:1440]
            df3 = a[1458:2118] 
            df_majority_downsampled = resample(df1, 
                             replace=True,    # sample without replacement
                             n_samples=660,     # to match minority class
                             random_state=0)

            df_majority_downsampled = df_majority_downsampled.sort_index(ascending=True)
            #print('DF1 downsampled:',df_majority_downsampled)
            #print('Length of DF1', len(df_majority_downsampled))

            df_majority_downsampled2 = resample(df2, 
                             replace=True,    # sample without replacement
                             n_samples=660,     # to match minority class
                             random_state=45)
            #df_majority_downsampled2 = df_majority_downsampled2.reset_index(drop=True)
            df_majority_downsampled2 = df_majority_downsampled2.sort_index(ascending=True)
            #print('DF2 downsampled:',df_majority_downsampled2)
            #print('Length of DF2', len(df_majority_downsampled2))


            split.append(df_majority_downsampled )
            split.append(df_majority_downsampled2)
            split.append(df3)
            fileorigins.append(name)
            #print(len(df_majority_downsampled),len(df_majority_downsampled2),len(df3))
            #plt.figure(figsize = (15,10))

            #ax = df_majority_downsampled.plot(linewidth = 1, figsize=(15,10))
            #ax.set_ylim(-2,10)


    return split

#Function for extracting your labels 
def extract_labels(labelArray):
    lbl=[]
    for weld in labelArray:
        lbl.append(CLASSES[weld[0]])
    a = np.reshape(np.array(lbl),(-1,1))
    return a

#Function for reshaping your training and testing datasets
def reshape_train_test(data):
    x = [a[['P-Raw','Temp','R-Raw']].to_numpy() for a in data]
    y = np.array(x)
    z = np.reshape(y,(-1,660,3))
    return z


# In[5]:
X_train = preprocess_train_test(root_train)
N_CLASSES = 2
CLASSES = {"Bad":0,"Good":1} 
label_for_train = [a["classification"].to_numpy()for a in X_train ]
y_train = extract_labels(label_for_train)
X_train1 = reshape_train_test(X_train)

# In[6]:
y_train

# In[10]:

from tensorflow.keras.optimizers import Adam,SGD,RMSprop
model = Sequential()
model.add(LSTM(128,input_shape=(660,3),return_sequences=True))#True = many to many
model.add(Dropout(0.2))
model.add(LSTM(128,input_shape=(660,3),return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128,input_shape=(660,3),return_sequences=False))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer ='Adam',metrics=['accuracy'])


# In[11]:


start_time = datetime.now()
print('The execution started on {}'.format(start_time))
fit_history = model.fit(X_train1,y_train,epochs=10,batch_size=16,validation_split = 0.2,verbose=2)
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# In[12]:


plt.figure(1, figsize = (15,8)) 
plt.rcParams.update({'font.size': 15})
plt.subplot(221)  
plt.plot(fit_history.history['acc'])  
plt.plot(fit_history.history['val_acc'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'valid'])
#plt.savefig('C:/Users/nadhola/Desktop/Graph for three parameter/threeparameterwithsampling660/1msesigmoidadamaccuracy.png', bbox_inches='tight')
    
plt.subplot(222)  
plt.plot(fit_history.history['loss'])  
plt.plot(fit_history.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'valid']) 
#plt.savefig('C:/Users/nadhola/Desktop/Graph for three parameter/threeparameterwithsampling660/3binarycrosssigmoidadamlr.png', bbox_inches='tight')
plt.show()


# In[ ]:





# In[16]:


#root_test = r'C:\Users\sanatara\DeepLearning\Scripts\WeldingInspection\TimeSeries\data\20190725_Precitec\AR3000_Stihl\Defocus\Test'
root_test =r'C:\Users\sanatara\DeepLearning\Scripts\WeldingInspection\TimeSeries\data\20190725_Precitec\DataSet_Felix'
X_test = preprocess_train_test(root_test)
X_test1 = reshape_train_test(X_test)
label_for_test = [a["classification"].to_numpy()for a in X_test]
y_test = extract_labels(label_for_test)
predictions = model.predict_classes(X_test1, batch_size=1, verbose=1)
print(len(X_test1))
print(len(y_test))


# In[ ]:





# In[17]:


predictions_reshaped = np.reshape(predictions, (-1,3))


# In[ ]:


#unique, counts = np.unique(predictions_reshaped[])


# In[18]:


thing = zip(fileorigins,predictions_reshaped)


# In[ ]:





# In[19]:


for fileorigin , pred in thing:
    print(fileorigin,end="    ")
    print(pred)


# In[ ]:


#print(len(X_test1))
#print(len(y_test))


# In[ ]:


#X_test1


# In[20]:


print(predictions)


# In[ ]:


print('Accuracy Score:', accuracy_score(y_test, predictions))


# In[1]:


cm = confusion_matrix(y_test, predictions)
import itertools
def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion Matrix', cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized Confusion Matrix")
    else:
        print('Confusion Matrix, without normalization')
        
    print(cm)
    
    thresh = cm.max() / 2.
    for i, j  in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
        
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
#    plt.savefig('C:/Users/nadhola/Desktop/Graph for three parameter/threeparameterwithsampling660/cf3binarysigmoidadamlr.png', bbox_inches='tight')

cm_plot_labels = ['Good','Bad']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')


# In[ ]:




