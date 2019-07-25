import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops
ops.reset_default_graph()
import pandas as pd
  

print("Test")
#matplotlib inline
from tqdm import tqdm
import os
root ="C:\\Users\sanatara\\DeepLearning\\Scripts\\WeldingInspection\\TimeSeries\\data"
allfiles =[]
pandaframes=[]
numClasses = 2
numInputDataColumns = 4

with tqdm(total=136) as pbar:
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
            pandaframes.append(frame)
            pbar.update()
pbar.close()

print("All done importing")

tf.set_random_seed(7)
sess = tf.Session()

result = pd.concat(pandaframes, ignore_index=True, )

result.columns = result.columns.str.replace("[+]", "plus")

result.classification= pd.factorize(result.classification)[0]

result.drop(['P-Refplus','Plasma','Error','P-Ref-','T-Refplus','T-Ref-','T-Raw','R-Refplus','R-Refplus','Refl','R-Ref-','IDMplus','IDM','IDM-','L-Raw','LP-Raw','LPplus','Power','LP-','classification'],axis=1).to_numpy()

x_vals = np.array([[x[0:numInputDataColumns]] for x in result.drop('classification', axis=1).to_numpy()])
x_vals = x_vals.reshape((x_vals.shape[0],numInputDataColumns))
y_vals1 = np.array([1 if y==0 else -1 for y in result.classification.values])
y_vals2 = np.array([1 if y==1 else -1 for y in result.classification.values])

batch_size = 10
#y_vals3 = np.array([1 if y==2 else -1 for y in result.classification.as_matrix()])
#y_vals4 = np.array([1 if y==3 else -1 for y in result.classification.as_matrix()])
#y_vals5 = np.array([1 if y==4 else -1 for y in result.classification.as_matrix()])
#y_vals6 = np.array([1 if y==5 else -1 for y in result.classification.as_matrix()])
#y_vals7 = np.array([1 if y==6 else -1 for y in result.classification.as_matrix()])
#y_vals8 = np.array([1 if y==7 else -1 for y in result.classification.as_matrix()])
#y_vals = np.array([y_vals1, y_vals2, y_vals3,y_vals4, y_vals5, y_vals6,y_vals7, y_vals8])

y_vals = np.array([y_vals1, y_vals2])
class1_x = [x[0] for i,x in enumerate(x_vals) if
result.classification[i]==0]
class1_y = [x[1] for i,x in enumerate(x_vals) if
result.classification[i]==0]
class2_x = [x[0] for i,x in enumerate(x_vals) if
result.classification[i]==1]
class2_y = [x[1] for i,x in enumerate(x_vals) if
result.classification[i]==1]
#class3_x = [x[0] for i,x in enumerate(x_vals) if
#result.classification[i]==2]
#class3_y = [x[1] for i,x in enumerate(x_vals) if
#result.classification[i]==2]
#class4_x = [x[0] for i,x in enumerate(x_vals) if
#result.classification[i]==3]
#class4_y = [x[1] for i,x in enumerate(x_vals) if
#result.classification[i]==3]
#class5_x = [x[0] for i,x in enumerate(x_vals) if
#result.classification[i]==4]
#class5_y = [x[1] for i,x in enumerate(x_vals) if
#result.classification[i]==4]
#class6_x = [x[0] for i,x in enumerate(x_vals) if
#result.classification[i]==5]
#class6_y = [x[1] for i,x in enumerate(x_vals) if
#result.classification[i]==5]
#class7_x = [x[0] for i,x in enumerate(x_vals) if
#result.classification[i]==6]
#class7_y = [x[1] for i,x in enumerate(x_vals) if
#result.classification[i]==6]
#class8_x = [x[0] for i,x in enumerate(x_vals) if
#result.classification[i]==7]
#class8_y = [x[1] for i,x in enumerate(x_vals) if
#result.classification[i]==7]
#
## Initialize placeholders
x_data = tf.placeholder(shape=[None, numInputDataColumns], dtype=tf.float32)
y_target = tf.placeholder(shape=[numClasses, None], dtype=tf.float32)
prediction_grid = tf.placeholder(shape=[None, numInputDataColumns], dtype=tf.float32)
#
## Create variables for svm
b = tf.Variable(tf.random_normal(shape=[numClasses,batch_size]))

print("Variables for SVM initialised")
#
## Gaussian (RBF) kernel
gamma = tf.constant(-10.0)
dist = tf.reduce_sum(tf.square(x_data), 1)
dist = tf.reshape(dist, [-1,1])
sq_dists = tf.multiply(4., tf.matmul(x_data, tf.transpose(x_data)))
my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))

# Declare function to do reshape/batch multiplication
def reshape_matmul(mat, _size):
    v1 = tf.expand_dims(mat, 1)
    v2 = tf.reshape(v1, [numClasses, _size, 1])
    return(tf.matmul(v2, v1))
    
# Compute SVM Model
first_term = tf.reduce_sum(b)
b_vec_cross = tf.matmul(tf.transpose(b), b)
y_target_cross = reshape_matmul(y_target, batch_size)

second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(b_vec_cross, y_target_cross)),[1,2])
loss = tf.reduce_sum(tf.negative(tf.subtract(first_term, second_term)))

# Gaussian (RBF) prediction kernel
rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1),[-1,1])
rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1),[-1,1])
pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(4., tf.matmul(x_data, tf.transpose(prediction_grid)))), tf.transpose(rB))
pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))

prediction_output = tf.matmul(tf.multiply(y_target,b), pred_kernel)
prediction = tf.argmax(prediction_output-tf.expand_dims(tf.reduce_mean(prediction_output,1), 1), 0)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(y_target,0)), tf.float32))

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.0001)
train_step = my_opt.minimize(loss)


# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Training loop
loss_vec = []
batch_accuracy = []
for i in range(1000):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = x_vals[rand_index]
    rand_y = y_vals[:,rand_index]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)
    
    acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x,
                                             y_target: rand_y,
                                             prediction_grid:rand_x})
    batch_accuracy.append(acc_temp)
    
    if (i+1)%25==0:
        print('Step #' + str(i+1))
        print('Loss = ' + str(temp_loss))
        

# Plot batch accuracy
plt.plot(batch_accuracy, 'k-', label='Accuracy')
plt.title('Batch Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Plot loss over time
plt.plot(loss_vec, 'k-')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()
