#!/usr/bin/env python
# coding: utf-8

# In[2]:


# TRAIN = "Train\\"
# TEST = "Test\\"
# #print(TRAIN)

# # Load "X" (the neural network's training and testing inputs)
# print(TRAIN)
# print(TEST)
# print("All done importing")
#         #print(pandaframes)
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm import tqdm
import tensorflow as tf
import seaborn as sns
from tensorflow.contrib import rnn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import os
from scipy import stats 
import seaborn as sns # used for plot interactive graph.
root ="C:\\Users\\sanatara\\DeepLearning\\Scripts\\WeldingInspection\\TimeSeries\\data\\20190704_Precitec\\Dummy"

from random import seed
from random import sample

#C:\Users\nadhola\Desktop\new2\data\Train\All good
allfiles =[]
#pandaframes=[]
X_train=[]

def add_row(df, row):
    df.loc[-1] = row
    df.index = df.index + 1  
    return df.sort_index()

def downsample(df, numSamples):
    majorityClassSize = len(df)
    samplesToIgnore = majorityClassSize - numSamples
    print(samplesToIgnore)
    exclusionRegion = 150
    startIndex = exclusionRegion
    endIndex = majorityClassSize - startIndex
#    samplingFrequency = (endIndex - startIndex) / samplesToIgnore 
#    print(a[0:a.size:3])
#    majorityClass = df.to_numpy()
#    majorityClass[startIndex:endIndex]
    seed(1)
    # prepare a sequence
    sequence = [i for i in range(startIndex, endIndex, samplesToIgnore)]
    print(sequence)
    # select a subset without replacement
#    subset = sample(sequence, 5)
#    print(subset)

    
    

with tqdm(total=128) as pbar:
    for root, dirs, files in os.walk(root):
        data = [os.path.join(root,f) for f in files if f.endswith(".txt")]
        #print(dirs)
        for file in data:
            dest = file.split("\\")  
            classification = dest[-2]
            s = "\\\\"
            dest= s.join(dest)
            frame = pd.read_csv(str(dest), delim_whitespace=True,header=0, index_col=0,skiprows=9)
            frame['classification']= classification
            frame=frame.loc[:, ~frame.columns.str.replace("(\.\d+)$", "").duplicated()]
            df = pd.DataFrame(data = frame)
            dataFrame = df.drop(['Time','P-Ref+','Plasma','Error','P-Ref-','P-MW','T-Ref+','T-Ref-','T-MW','T-Raw','R-Ref+','Refl','R-Ref-','R-MW'],axis=1)
            
            df1 = dataFrame[0:721]
            downsample(df1, 600)
            X_train.append(df1)
            print (len(df1))
            
            
            #list2=[]
            df2 = dataFrame[745:1443]
            print (len(df2))
            X_train.append(df2)
            #list3=[]
            df3 =[]
            df3 = dataFrame[1458:2118]
            
            
            
            fig, ax = plt.subplots(num=None, figsize=(14, 6), dpi=80, facecolor='w', edgecolor='k')
            size = len(df1)
            #ax.set_ylim(0,energy.max())
            pRaw = df1['P-Raw'].to_numpy()
            print(pRaw)
            ax.plot(range(0,size), pRaw, '‑', color='blue', animated = True, linewidth=1)
#            ax.plot(range(0,size), df1['Temp'], '‑', color='red', animated = True, linewidth=1)
#            ax.plot(range(0,size), df1['R-Raw'], '‑', color='green', animated = True, linewidth=1)


#            fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
#
#            df1.plot(kind='line',y='P-Raw',  color='red', ax=ax1)
#            df2.plot(kind='line',y='P-Raw',  color='green', ax=ax1)
#            df3.plot(kind='line',y='P-Raw',  color='blue', ax=ax1)
#                        
#            df1.plot(kind='line',y='Temp',  color='green', ax=ax2)
#            df1.plot(kind='line',y='R-Raw',  color='blue', ax=ax3)
                        
#            ax[0, 0].plot(range(10), 'r') #row=0, col=0
#            ax[1, 0].plot(range(10), 'b') #row=1, col=0
#            ax[0, 1].plot(range(10), 'g') #row=0, col=1
#            ax[1, 1].plot(range(10), 'k') #row=1, col=1
            plt.show()

#            df4 = dataFrame[2112:2118]
#            df3 = df3.append(df4)
            
#            ax = plt.gca()
#
#            df1.plot(kind='line',y='P-Raw',  color='red', ax=ax)
#            df1.plot(kind='line',y='Temp',  color='blue', ax=ax)
#            df1.plot(kind='line',y='R-Raw',  color='green', ax=ax)
#            plt.show(df1)
#                        
#            
#            df1.plot(kind='line',y='P-Raw',  color='red', ax=ax)
#            df1.plot(kind='line',y='Temp',  color='blue', ax=ax)
#            df1.plot(kind='line',y='R-Raw',  color='green', ax=ax)
            
#            df3.index = range(1416:2118+6)
#            add_row(df3, df4)            
            print(len(df3))
            print(df3)
#            print(len(df4))
#            print (df4)
            X_train.append(df3)
#            print(len(X_train))
            #dataArray = [a[['P-Raw','Temp','R-Raw']].to_numpy() for a in X_train]
            #data = np.array(dataArray )
            #list2=pd.concat([df1,df2,df3]).drop_duplicates(keep=False)
            #pandaframes.append(frame)
            #dff=pd.DataFrame(df3)
            #print(list2)
            #print(X_train)
            pbar.update()
pbar.close()

print("All done importing")
        #print(pandaframes)
            


## In[15]:
#
#
#dataArray1 = [a[['P-Raw','Temp','R-Raw']].to_numpy() for a in X_train]
##ts = np.concatenate(dataArray2)
#data1 = np.array(dataArray1)
#data1 = np.reshape(data1,(-1,706,3))
#
#
## In[8]:
#
#
#X_train1 = data1
##print(data1 )
#print(X_train1.shape)
#print(X_train1[0].shape)
#

# In[16]:


root ="C:\\Users\\sanatara\\DeepLearning\\Scripts\\WeldingInspection\\TimeSeries\\data\\20190704_Precitec\\data\\Test"
##C:\Users\nadhola\Desktop\new2\data\Train\All good
#allfiles =[]
##pandaframes=[]
#X_test=[]
#
#with tqdm(total=128) as pbar:
#    for root, dirs, files in os.walk(root):
#        data = [os.path.join(root,f) for f in files if f.endswith(".txt")]
#        #print(dirs)
#        for file in data:
#            dest = file.split("\\")  
#            classification = dest[-2]
#            s = "\\\\"
#            dest= s.join(dest)
#            frame = pd.read_csv(str(dest), delim_whitespace=True,header=0, index_col=0,skiprows=9)
#            frame['classification']= classification
#            frame=frame.loc[:, ~frame.columns.str.replace("(\.\d+)$", "").duplicated()]
#            df = pd.DataFrame(data = frame)
#            naresh = df.drop(['Time','P-Ref+','Plasma','Error','P-Ref-','P-MW','T-Ref+','T-Ref-','T-MW','T-Raw','R-Ref+','Refl','R-Ref-','R-MW'],axis=1)
#            df1 = naresh[0:706]
#            X_test.append(df1)
#            #list2=[]
#            df2 = naresh[706:1412]
#            X_test.append(df2)
#            #list3=[]
#            df3 = naresh[1412:2118]
#            X_test.append(df3)
#            #list2=pd.concat([df1,df2,df3]).drop_duplicates(keep=False)
#            #pandaframes.append(frame)
#            #dff=pd.DataFrame(df3)
#            #print(list2)
#            #print(X_test)
#            #dataArray = [a[['P-Raw','Temp','R-Raw']].to_numpy() for a in X_test]
#            #data = np.array(dataArray )
#            #print(len(X_test))
#            pbar.update()            
#            
#            
#pbar.close()
#
#print("All done importing")

#
## In[ ]:
#
#
##from tsfresh.examples import load_robot_execution_failures
##from tsfresh import extract_features
##X_train1, _ = load_robot_execution_failures()
##X = extract_features(X_train1, column_id='P-Raw', column_sort='Temp', column_name='R-Raw')
##print(X_train1)
#
#
## In[17]:
#
#
#dataArray2 = [a[['P-Raw','Temp','R-Raw']].to_numpy() for a in X_test]
##ts = np.concatenate(dataArray2)
#data2 = np.array(dataArray2)
#data2 = np.reshape(data2,(-1,706,3))
#
#
## In[19]:
#
#
#X_test1 = data2
##print(X_test1)
#
#
## In[20]:
#
#
#N_CLASSES = 2
#CLASSES = {"Bad":0,
#          "Good":1} 
#labelArray1 = [a["classification"].to_numpy()for a in X_train ]
#labelArray2 = [a["classification"].to_numpy()for a in X_test ]
#
#
## In[25]:
#
#
#numberLabelArray1=[]
#for weld in labelArray1:
#    numberLabel=[]
##     for timestep in weld:
##         labelnumber = CLASSES[timestep]
##         numberLabel.append(labelnumber)
##     numberLabelArray1.append(numberLabel)
#    numberLabelArray1.append(CLASSES[weld[0]])
#
#y_train = np.reshape(np.array(numberLabelArray1),(-1,1))
#
#
## In[26]:
#
#
#numberLabelArray=[]
#for weld in labelArray2:
#    numberLabel=[]
#    #for timestep in weld:
##         labelnumber = CLASSES[timestep]
##         numberLabel.append(labelnumber)
##     numberLabelArray2.append(numberLabel)
#    numberLabelArray.append(CLASSES[weld[0]])
#
#y_test = np.reshape(np.array(numberLabelArray),(-1,1))
#
#
## In[ ]:
#
#
##len(labelArray1)
#
#
## In[ ]:
#
#
##from keras.preprocessing.sequence import TimeseriesGenerator
#
##generator = TimeseriesGenerator(X_train1, X_train1, length =3,  batch_size = 1)
#
#
## In[22]:
#
#
#import matplotlib.pyplot as plt
#from keras.models import Sequential
#from keras.layers import LSTM, Dense, Activation
#model = Sequential()
#model.add(LSTM(8,input_shape=(706,3),return_sequences=False))#True = many to many
#model.add(Dense(2,kernel_initializer='normal',activation='linear'))
#model.add(Dense(1,kernel_initializer='normal',activation='linear'))
#model.compile(loss='mse',optimizer ='adam',metrics=['accuracy'])
#
#
## In[ ]:
#
#
##model.fit_generator(generator,labelArray1,epochs=2000,verbose=1);
#model.fit(X_train1,y_train,epochs=2000,batch_size=5,validation_split=0.05,verbose=0);
#
#
## In[ ]:
#
#
#scores = model.evaluate(X_train1,y_train,verbose=1,batch_size=5)
#print('Accurracy: {}'.format(scores[1]))
#
#
#
## In[ ]:
#
#
#predict=model.predict(X_test1)
#plt.plot(y_test, predict, 'C2')
#plt.ylim(ymax = 3, ymin = -3)
#plt.show()

