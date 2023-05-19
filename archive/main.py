#!/usr/bin/env python
# coding: utf-8

# This notebook is an attempt to predict bone age using Xception(pre trained model)<br>
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import datetime, os
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print("# In[1]:")


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in



# **Some Setup**<br>
# The cell below creates the pandas dataframes for training and testing.

print("# In[2]:")


#loading dataframes
train_df = pd.read_csv('boneage-training-dataset.csv')
test_df = pd.read_csv('boneage-test-dataset.csv')

#appending file extension to id column for both training and testing dataframes
train_df['id'] = train_df['id'].apply(lambda x: str(x)+'.png')
test_df['Case ID'] = test_df['Case ID'].apply(lambda x: str(x)+'.png')

train_df.head()


# **Some EDA and feature engineering follow**

print("# In[3]:")


#finding out the number of male and female children in the dataset
#creating a new column called gender to keep the gender of the child as a string
train_df['gender'] = train_df['male'].apply(lambda x: 'male' if x else 'female')
print(train_df['gender'].value_counts())
sns.countplot(x = train_df['gender'])


print("# In[4]:")


#oldest child in the dataset
print('MAX age: ' + str(train_df['boneage'].max()) + ' months')

#youngest child in the dataset
print('MIN age: ' + str(train_df['boneage'].min()) + ' months')

#mean age is
mean_bone_age = train_df['boneage'].mean()
print('mean: ' + str(mean_bone_age))

#median bone age
print('median: ' +str(train_df['boneage'].median()))

#standard deviation of boneage
std_bone_age = train_df['boneage'].std()

#models perform better when features are normalised to have zero mean and unity standard deviation
#using z score for the training
train_df['bone_age_z'] = (train_df['boneage'] - mean_bone_age)/(std_bone_age)

print(train_df.head())


print("# In[5]:")


#plotting a histogram for bone ages
train_df['boneage'].hist(color = 'green')
# plt.xlabel('Age in months')
# plt.ylabel('Number of children')
# plt.title('Number of children in each age group')


print("# In[6]:")


train_df['bone_age_z'].hist(color = 'violet')
# plt.xlabel('bone age z score')
# plt.ylabel('Number of children')
# plt.title('Relationship between number of children and bone age z score')


print("# In[7]:")


#Relationship between age and gender with a categorical scatter plot (swarmplot)
# sns.swarmplot(x = train_df['gender'], y = train_df['boneage'])


print("# In[8]:")


#distribution of age within each gender
male = train_df[train_df['gender'] == 'male']
female = train_df[train_df['gender'] == 'female']
# fig, ax = plt.subplots(2,1)
# ax[0].hist(male['boneage'], color = 'blue')
# ax[0].set_ylabel('Number of boys')
# ax[1].hist(female['boneage'], color = 'red')
# ax[1].set_xlabel('Age in months')
# ax[1].set_ylabel('Number of girls')
# fig.set_size_inches((10,7))


print("# In[9]:")


#splitting train dataframe into traininng and validation dataframes
train_df['boneage_category'] = pd.cut(train_df['boneage'], 10)
df_train, df_valid = train_test_split(train_df, test_size = 0.2, random_state = 0)


# ## Normalize the data in train set

print("# In[10]:")


# df_train = df_train1.groupby(['boneage_category', 'male']).apply(lambda x: x.sample(500, replace = True)
#                                                       ).reset_index(drop = True)
# print('New Data Size:', df_train.shape[0], 'Old Size:', df_train1.shape[0])
# train_df[['boneage', 'gender']].hist(figsize = (10, 5))


# Looking into the dataset...

print("# In[11]:")


import matplotlib.image as mpimg
# for filename, boneage, gender in train_df[['id','boneage','gender']].sample(4).values:
#     images = mpimg.imread('boneage-training-dataset/'+ filename)
#     plt.imshow(images)
#     plt.title('Image name:{}  Bone age: {} years  Gender: {}'.format(filename, boneage/12, gender))
#     plt.axis('off')
#     plt.show()


# **Setting up Image Data Generators!**<br>
# We use image data generators for both training, testing and preprocessing of images. Validation set is already broken off from training set.

print("# In[12]:")


#library required for image preprocessing
from keras.preprocessing.image import ImageDataGenerator
from  keras.applications.xception import preprocess_input

#reducing down the size of the image
img_size = 299
data_augmenation = dict(rotation_range=0.2, zoom_range=0.1, horizontal_flip=True,
                                width_shift_range=0.05,
                                height_shift_range=0.05,
                                shear_range=0.05, fill_mode='nearest')
train_data_generator = ImageDataGenerator(preprocessing_function = preprocess_input,  **data_augmenation)
val_data_generator = ImageDataGenerator(preprocessing_function = preprocess_input)

#train data generator
train_generator = train_data_generator.flow_from_dataframe(
    dataframe = df_train,
    directory = 'boneage-training-dataset/boneage-training-dataset',
    x_col= 'id',
    y_col= 'bone_age_z',
    batch_size = 32,
    seed = 42,
    shuffle = True,
    class_mode= 'other',
    flip_vertical = True,
    color_mode = 'rgb',
    target_size = (img_size, img_size))

#validation data generator
val_generator = val_data_generator.flow_from_dataframe(
    dataframe = df_valid,
    directory = 'boneage-training-dataset/boneage-training-dataset',
    x_col = 'id',
    y_col = 'bone_age_z',
    batch_size = 32,
    seed = 42,
    shuffle = True,
    class_mode = 'other',
    flip_vertical = True,
    color_mode = 'rgb',
    target_size = (img_size, img_size))

#test data generator
test_data_generator = ImageDataGenerator(preprocessing_function = preprocess_input)

test_generator = test_data_generator.flow_from_directory(
    directory = 'boneage-test-dataset',
    shuffle = True,
    class_mode = None,
    color_mode = 'rgb',
    target_size = (img_size,img_size))


print("# In[13]:")


test_X, test_Y = next(val_data_generator.flow_from_dataframe(
                            df_valid,
                            directory = 'boneage-training-dataset/boneage-training-dataset',
                            x_col = 'id',
                            y_col = 'bone_age_z',
                            target_size = (img_size, img_size),
                            batch_size =  2300,
                            class_mode = 'other'
                            ))


#  The function to plot training and validation error as a function of epochs

print("# In[14]:")


def plot_it(history):
    '''function to plot training and validation error'''
    fig, ax = plt.subplots( figsize=(20,10))
    ax.plot(history.history['mae_in_months'])
    ax.plot(history.history['val_mae_in_months'])
    plt.title('Model Error')
    plt.ylabel('error')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    ax.grid(color='black')
    plt.show()


print("# In[15]:")


from keras.metrics import mean_absolute_error
def mae_in_months(x_p, y_p):
    '''function to return mae in months'''
    return mean_absolute_error((std_bone_age*x_p + mean_bone_age), (std_bone_age*y_p + mean_bone_age))


print("# In[16]:")


from keras.layers import GlobalMaxPooling2D, Dense,Flatten, Dropout
from keras.callbacks import TensorBoard, ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from keras import Sequential
import tensorflow as tf
# TODO changed
# import keras.applications.resnet50
from keras.api._v2.keras.applications.resnet50 import ResNet50


# model_1 = ResNet50(input_shape = (img_size, img_size, 3),
#                                            include_top = False,
#                                            weights = 'imagenet')
# model_1 = tf.keras.applications.xception.Xception(input_shape = (img_size, img_size, 3),
#                                            include_top = False,
#                                            weights = 'imagenet')
# model_1 = tf.keras.applications.InceptionResNetV2(input_shape = (img_size, img_size, 3),
#                                            include_top = False,
#                                            weights = 'imagenet')
##******************************
# model_1 = tf.keras.applications.ResNet152V2(input_shape = (img_size, img_size, 3),
#                                            include_top = False,
#                                            weights = 'imagenet')
# for i, layer in enumerate(model_1.layers[:]):
#     if i < 70:
#         layer.trainable = False
#     else:
#         layer.trainable = True
#     print(i, layer)
## ****************************************
model_1 = tf.keras.applications.DenseNet201(input_shape = (img_size, img_size, 3),
                                           include_top = False,
                                           weights = 'imagenet')
model_1.trainable = True
model_2 = Sequential()
model_2.add(model_1)
model_2.add(GlobalMaxPooling2D())
model_2.add(Flatten())
model_2.add(Dense(64, activation = 'relu'))
# model_2.add(Dropout(0.5))
model_2.add(Dense(32, activation = 'relu'))
model_2.add(Dense(1, activation = 'linear'))

Sgd = tf.keras.optimizers.SGD(
    learning_rate=0.001, momentum=0.0, nesterov=False)
NaDam = tf.keras.optimizers.Nadam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
AdaMax = tf.keras.optimizers.Adamax(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
#compile model
model_2.compile(loss ='mse', optimizer= AdaMax, metrics = [mae_in_months] )
# model_2.compile(loss ='mse', optimizer= 'adam', metrics = [mae_in_months] )

#model summary
model_2.summary()


print("# In[17]:")


# Load the TensorBoard notebook extension
# get_ipython().run_line_magic('load_ext', 'tensorboard')
logs_dir = '.\logs'
# get_ipython().run_line_magic('tensorboard', '--logdir {logs_dir}')


print("# In[18]:")


#early stopping
early_stopping = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience= 12,
                              verbose=0, mode='auto')

#model checkpoint
mc = ModelCheckpoint('resnet50_best_imsize299_model.h5', monitor='val_loss', mode='min', save_best_only=True,save_weights_only = False)
# mc = ModelCheckpoint('xception_best_imsize299_model2.h5', monitor='val_loss', mode='min', save_best_only=True,save_weights_only = False)

#tensorboard callback
logdir = os.path.join(logs_dir,datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tensorboard_callback =  TensorBoard(logdir, histogram_freq = 1)

#reduce lr on plateau
red_lr_plat = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

# callbacks = [tensorboard_callback,early_stopping,mc, red_lr_plat]
callbacks = [early_stopping,mc, red_lr_plat]


# TODO revert
# epochs = 60
# steps_per_epoch = 315
#fit model
# history = model_2.fit_generator(train_generator,
#                             steps_per_epoch = 315,
#                             validation_data = val_generator,
#                             validation_steps = 1,
#                             epochs = 30,
#                             callbacks= callbacks)
# history

# import pickle
# import sys
# sys.setrecursionlimit(10000)
#
# with open('filename.pickle', 'wb') as handle:
#     pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('filename.pickle', 'rb') as handle:
#     history = pickle.load(handle)

# get_ipython().run_line_magic('tensorboard', '--logdir logs')
# plot_it(history)


print("# In[19]:")


# for i, layer in enumerate(model_1.layers[:]):
#     if i < 300:
#         layer.trainable = False
#     print(i, layer)
#       # Check the trainable status of the individual layers
# for layer in model_1.layers:
#     print(layer, layer.trainable)


# Evaluating the best saved model on the validation data and visualising results!!

print("# In[20]:")


# from tensorflow import keras
# model_2 =  keras.models.load_model('resnet50_best_imsize299_model.h5')
model_2.load_weights('resnet50_best_imsize299_model.h5')
pred = mean_bone_age + std_bone_age*(model_2.predict(test_X, batch_size = 32, verbose = True))
test_months = mean_bone_age + std_bone_age*(test_Y)

ord_ind = np.argsort(test_Y)
ord_ind = ord_ind[np.linspace(0, len(ord_ind)-1, 12).astype(int)] # take 8 evenly spaced ones
fig, axs = plt.subplots(6, 2, figsize = (15, 30))
for (ind, ax) in zip(ord_ind, axs.flatten()):
    ax.imshow(test_X[ind, :,:,0], cmap = 'bone')
    ax.set_title('Age: %fY\nPredicted Age: %fY' % (test_months[ind]/12.0,
                                                           pred[ind]/12.0))
    ax.axis('off')
fig.savefig('trained_image_predictions.png', dpi = 300)

pred_reshape = np.reshape(pred,pred.shape[0])
print("mean of absolute difference for: ", test_months.shape[0], " cases in test set: ", np.mean(abs(pred_reshape-test_months)))


print("# In[21]:")


from sklearn.metrics import mean_absolute_error
mean_absolute_error(pred_reshape, test_months)


print("# In[22]:")


# fig, ax = plt.subplots(figsize = (7,7))
# ax.plot(test_months, pred, 'r.', label = 'predictions')
# ax.plot(test_months, test_months, 'b-', label = 'actual')
# ax.legend(loc = 'upper right')
# ax.set_xlabel('Actual Age (Months)')
# ax.set_ylabel('Predicted Age (Months)')


# **The plot deviates from the line at very old and very young ages probably because we have less examples for those cases in the dataset**

# Predicting on test data, we obtain:

print("# In[23]:")


test_generator.reset()
y_pred = model_2.predict_generator(test_generator)
predicted = y_pred.flatten()
predicted_months = mean_bone_age + std_bone_age*(predicted)
filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions": predicted_months})
with open(r"C:\Users\d\Desktop\University\Graduation project\archive\predicted_months.txt", "w") as file:
    file.write(str(predicted_months))
results.to_csv("xception_best_imsize299-results.csv",index=False)

print(results)


# assuming y_true and y_pred are the true and predicted labels, respectively.

acc = accuracy_score(predicted_months, test_months)


print("# In[24]:")


test_df


