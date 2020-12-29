#!/usr/bin/env python
# coding: utf-8

# # Fashion MNIST

# [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) can be used as drop-in replacement for the original MNIST dataset (10 categories of handwritten digits). It shares the same image size (28x28) and structure of training (60,000) and testing (10,000) splits. The class labels are:
# 
# | Label|	Description|
# |-|-|
# |0|	T-shirt/top|
# |1|	Trouser|
# |2|	Pullover|
# |3|	Dress|
# |4|	Coat|
# |5|	Sandal|
# |6|	Shirt|
# |7|	Sneaker|
# |8|	Bag|
# |9|	Ankle boot|
# 
# **Example**
# 
# <img src="https://github.com/zalandoresearch/fashion-mnist/raw/master/doc/img/fashion-mnist-sprite.png" width="50%"/>

# In this notebook, you need to train a classical ML model (no deep learning) to reach the highest accruracy score. Please follow the ML project checklist and make sure you organize your code well.
# 
# - **Expected Accuracy Score on the given test set**: >89%
# - **Expected Accuracy Score on the HIDDEN test set**: as high as possible. Top 5 will be picked to present
# 
# **Submission:** 
# - Share your notebook to instructors (quan.tran@coderschool.vn), and prepare your presentation on the next Monday afternoon. 
# 
# - The submission.csv file. You can put them inside the submissions folder.
# The name of the file should be like this: \<your_name\>_submission.csv. For example: quantran_submission.csv
# 
# 
# **Extra optional requirements**:
# - Tuning your hyperparameters with both RandomSearch and GridSearch
# - Use Sklearn Pipeline (use California House Pricing notebook as an example)
# - Confusion Matrix
# - Plot the images that the model predicts incorrectly
# - Use confusion matrix and images plotted incorrectly to do error analysis

# ## Sample code to get fashion MNIST data

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift
get_ipython().run_line_magic('matplotlib', 'inline')

import pprint
pp = pprint.PrettyPrinter(indent=4)


import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")


# In[32]:


from tqdm import tqdm


# In[2]:


from tensorflow.keras.datasets import fashion_mnist

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

print('Training data:', X_train.shape, y_train.shape)
print('Test data:', X_test.shape, y_test.shape)


# # Sample code to display images

# In[4]:


X_train[3].shape, y_train[3]


# In[5]:


X_train[3]


# In[6]:


print('Label:', y_train[3])
plt.imshow(X_train[3], cmap='gray')


# In[7]:


def shift_pixels(src, x, y):
    """
    shift x pixels vertically and y pixels horizontally
    """
    output = shift(src, [x,y], cval=0)
    
    return output


# In[8]:


plt.imshow(shift_pixels(X_train[3],0,3), cmap='gray')


# In[9]:


def denoise(src_img, threshold=50, display=False):
    #src_img = np.float32(src_img)
    #dst_img = cv.medianBlur(src_img, threshold)
    
    new_image = np.minimum(src_img, 255)
    new_image[new_image < threshold] = 0
    if display:
        plt.figure(figsize=(15,15))
        plt.subplot(121),plt.imshow(src_img , cmap='gray')
        plt.subplot(122),plt.imshow(new_image , cmap='gray')
    return new_image


# In[10]:


_ = denoise(X_train[3], display=True)


# In[11]:


def plot_images(images, labels=None, correct_labels=None):
    '''Plot images with their labels. Ten each row'''
    plt.figure(figsize=(20,20))
    columns = 10
    for i, image in enumerate(images):
        ax = plt.subplot(len(images) / columns + 1, columns, i + 1)
        if (not labels is None) and (not correct_labels is None):
            ax.set_title(f"Wr: {labels[i]} Ri: {correct_labels[i]}", fontsize=14)
        elif not labels is None:
            ax.set_title(f"{labels[i]}", fontsize=16)
        
        plt.axis('off')
        plt.subplots_adjust(bottom=0.1)
        plt.imshow(image, cmap='gray')


# In[12]:


def get_samples(n_samples, X, y=None):
    '''Get n_samples randomly'''
    samples_index = np.random.choice(np.arange(len(X)), n_samples, replace=False)
    if not y is None:
        return X[samples_index], y[samples_index]
    return X[samples_index]


# In[13]:


images, labels = get_samples(50, X_train, y_train)
plot_images(images, labels)


# # Your Code 

# These are numpy arrays:
# - X_train 
# - y_train 
# - X_test 
# - y_test

# In[14]:


print('Trainingm data:', X_train.shape, y_train.shape)
print('Test data:', X_test.shape, y_test.shape)


# Check the distribution of labels in both train and test. Looks balance!

# In[15]:


print('unique values in y_train:', np.unique(y_train, return_counts=True) )
print('unique values in y_test:', np.unique(y_test, return_counts=True) )


# In[16]:


sns.countplot(y_train)


# #### Pre-process data:
# - Scale pixel values
# - Augment the data: shift the images, denoise the images
# - Flatten the 2D array --> 1D vector
# 

# 

# In[28]:


from sklearn.base import BaseEstimator, TransformerMixin

class DenoiseTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        #return np.array([skimage.color.rgb2gray(img) for img in X])
        return np.array([denoise(img) for img in X])


# In[39]:


class ShiftTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, y):
        self.y = y
        pass
    
    def fit(self, X, y=None):
        return self
    
    def image_generator(self, src_image, src_label):
        # create placeholder
        output_image = np.zeros((9, *src_image.shape))

        output_image[0] = shift_pixels(src_image, 1, 0)
        output_image[1] = shift_pixels(src_image, -1, 0)
        output_image[2] = shift_pixels(src_image, 2, 0)
        output_image[3] = shift_pixels(src_image, -2, 0)
        output_image[4] = shift_pixels(src_image, 0, 1)
        output_image[5] = shift_pixels(src_image, 0, -1)
        output_image[6] = shift_pixels(src_image, 0, 2)
        output_image[7] = shift_pixels(src_image, 0, -2)
        output_image[8] = shift_pixels(src_image, 0, 0)
        output_label = np.array([src_label for i in range(9)])

        return output_image, output_label
    
    def transform(self, X, y=None):
        #print(y)
        X_out = np.zeros((36000, 28, 28))
        y_out = np.zeros((36000, ))
        for i in tqdm(range(X.shape[0])):
            X_out[i*9: i*9+9], y_out[i*9:i*9+9] = self.image_generator(X[i], self.y[i])
            
        return X_out, y_out


# In[40]:


n_samples = 4000
n_test_samples = 800
X_train_sample, y_train_flat = get_samples(n_samples, X_train, y_train)
X_test_sample, y_test_flat = get_samples(n_test_samples, X_test, y_test)
print('Original shape:', X_train_sample.shape, X_test_sample.shape)
# Normalization
X_train_flat = X_train_sample.reshape((n_samples, -1))/255
X_test_flat = X_test_sample.reshape((n_test_samples, -1))/255

print('Training data', X_train_flat.shape)
print('Test data', X_test_flat.shape)


# In[41]:


denoise_step = DenoiseTransformer()
shift_step = ShiftTransformer(y_train_flat)

denoised_X = denoise_step.fit_transform(X_train_sample)
new_X_train, new_y_train = shift_step.fit_transform(denoised_X) 

print(new_X_train.shape, new_y_train.shape)


# In[ ]:


# Normalization
X_train_flat = new_X_train.reshape((-1, 784)) / 255
X_test_flat = X_test_denoised.reshape((-1, 784))/255


# #### ImageGenerator
# - To increase different positions of in the training set, new images where object is shifted up, down, left, right 1 - 2 pixels are added. (8 more images are created per original image)

# In[ ]:


from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# In[ ]:


enhanced_train_data.shape, enhanced_train_label.shape


# In[ ]:


np.unique(enhanced_train_label, return_counts=True)


# In[ ]:


enhanced_train_data.shape, X_test_denoised.shape, y_test_flat.shape


# In[ ]:


model_SVC = SVC(C=10)

# Normalization
X_train_flat = enhanced_train_data.reshape((36000, 784))/255
#X_test_flat = X_test_sample.reshape((n_test_samples, -1))/255
X_test_flat = X_test_denoised.reshape((-1, 784))/255
model_SVC.fit( X_train_flat, enhanced_train_label)

y_pred = model_SVC.predict(X_test_flat)
print("Test accuracy:", accuracy_score(y_test_flat, y_pred))


# In[ ]:


from sklearn.metrics import plot_confusion_matrix
plt.figure(figsize=(20,20))
plot_confusion_matrix(model_SVC, X_test_flat, y_test_flat,
                                 cmap=plt.cm.Blues)


# #### Model training:
# - Linear SVM and Kernel SVM
# - Decision Tree
# - Random Forest
# - Boosting \& Bagging

# In[ ]:





# In[ ]:


model_SVC = SVC(C=3)

model_SVC.fit(X_train_flat, y_train_flat)
y_pred = model_SVC.predict(X_test_flat)


# In[ ]:


y_train_pred = model_SVC.predict(X_train_flat)
print("Train accuracy:", accuracy_score(y_train_flat, y_train_pred))


# In[ ]:


from sklearn.metrics import plot_confusion_matrix
plt.figure(figsize=(20,20))
plot_confusion_matrix(model_SVC, X_train_flat, y_train_flat,
                                 cmap=plt.cm.Blues)


# In[ ]:


# Your code here
wrong_predicted_images = X_train_flat[y_train_flat != y_train_pred].reshape((-1, 28, 28))
wrong_predictions = y_train_pred[y_train_flat != y_train_pred]
correct_label = y_train_flat[y_train_flat != y_train_pred]

plot_images(wrong_predicted_images, wrong_predictions, correct_label)


# In[ ]:


print("Test accuracy:", accuracy_score(y_test_flat, y_pred))


# In[ ]:


plt.figure(figsize=(20,20))
plot_confusion_matrix(model_SVC, X_test_flat, y_test_flat,
                                 cmap=plt.cm.Blues)


# In[ ]:


np.where(y_test_flat != y_pred) 


# In[ ]:


# Your code here
wrong_predicted_images = X_test_flat[y_test_flat != y_pred].reshape((-1, 28, 28))
wrong_predictions = y_pred[y_test_flat != y_pred]
correct_label = y_test_flat[y_test_flat != y_pred]

plot_images(wrong_predicted_images, wrong_predictions, correct_label)


# In[ ]:


# Use image generator to generate more training images




# In[ ]:


# # Use RandomizedSearchCV
# from sklearn.model_selection import RandomizedSearchCV

# parameters = {'gamma': np.arange(0.001, 10, 0.1),
#               'C': np.arange(0.001, 10, 0.1)}

# model = SVC(kernel='rbf')

# random_models = RandomizedSearchCV(estimator=model,
#                                    param_distributions=parameters,
#                                    n_iter=50,
#                                    scoring=make_scorer(accuracy_score),
#                                    cv=5, n_jobs=-1)
# random_models.fit(X_train_flat, y_train_flat)


# In[ ]:


#random_models.best_params_


# In[ ]:


#random_models.best_score_


# In[ ]:


# best_svm_model = random_models.best_estimator_
# svm_predictions = best_svm_model.predict(X_test_flat) # predictions on test set
# print('Accuracy Score:', accuracy_score(y_test_flat, svm_predictions))


# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


model_tree = DecisionTreeClassifier(max_depth=10, min_samples_leaf = 8,criterion='gini').fit(X_train_flat, y_train_flat)
accuracy_score(y_test_flat, model_tree.predict(X_test_flat) )


# In[ ]:


plot_confusion_matrix(model_tree, X_test_flat, y_test_flat,
                                 cmap=plt.cm.Blues)


# In[ ]:





# In[ ]:


# # USE THIS CELL TO TEST THE MIN_SAMPLES_LEAF OR MAX_DEPTH
# train_errors = []
# test_errors = []
# a_range = np.arange(2, 40)
# for i in a_range:
#     model_tree = DecisionTreeClassifier(max_depth=i, min_samples_leaf=8, criterion='gini').fit(X_train_flat, y_train_flat)
#     train_errors.append(1 - accuracy_score(y_train_flat, model_tree.predict(X_train_flat)))
#     test_errors.append(1 - accuracy_score(y_test_flat, model_tree.predict(X_test_flat)))

# plt.figure(figsize=(16, 9))
# plt.scatter(a_range, train_errors)
# plt.plot(a_range, train_errors, label='Training Error')
# plt.scatter(a_range, test_errors)
# plt.plot(a_range, test_errors, label='Test Error')
# plt.legend()
# plt.title('Decision Tree - Overfitting & Underfitting')
# plt.xlabel('Min sample leaf', fontsize=16)
# plt.ylabel('Error in %', fontsize=16)
# plt.show()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(n_estimators=40, max_depth=10, min_samples_leaf=8)
model_rf.fit(X_train_flat, y_train_flat)
accuracy_score(y_test_flat, model_rf.predict(X_test_flat) )


# In[ ]:


model_rf = RandomForestClassifier(n_estimators=40, max_depth=10, min_samples_leaf=8)
scores = cross_val_score(model_rf, X_train_flat, y_train_flat, cv=5)
scores


# In[ ]:





# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn import metrics

# Base learners
voting_classifiers = VotingClassifier( estimators=[('SVC', SVC(C=3)),
                    ('Decision Tree 10', DecisionTreeClassifier(max_depth=10, min_samples_leaf = 8,criterion='gini')),
                    ('RF', RandomForestClassifier(n_estimators=40, max_depth=10, min_samples_leaf=8)),
                    ('Logistic Regression', LogisticRegression(max_iter=2000))],
                      voting='hard')
voting_classifiers.fit(X_train_flat, y_train_flat)
accuracy_score(y_test_flat, voting_classifiers.predict(X_test_flat) )


# In[ ]:


from sklearn.ensemble import AdaBoostClassifier

model_adaboost = AdaBoostClassifier(n_estimators=200)
model_adaboost.fit(X_train_flat, y_train_flat)
accuracy_score(y_test_flat, model_adaboost.predict(X_test_flat) )


# In[ ]:


get_ipython().run_cell_magic('time', '', "from xgboost import XGBClassifier\n\nmodel_xgboost = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', max_depth=10)\nmodel_xgboost.fit(X_train_flat, y_train_flat)\nprint(accuracy_score(y_test_flat, model_xgboost.predict(X_test_flat) ))")


# In[ ]:





# In[ ]:





# # Test set
# 
# Here is the test set without label (FMNIST_augmented_test.npy). You will use your trained machine learning model to make predictions on this test set, and then submit a csv file containing the predictions 

# In[ ]:


#from google.colab import drive
#drive.mount('/content/gdrive')


# In[ ]:


#PATH = '/content/gdrive/MyDrive/FTMLE | 2020.11 | Izu/Week_6/Weekly_Project/'
PATH = './'


# In[ ]:


X_test_augmented = np.load(PATH + 'FMNIST_augmented_test.npy')


# In[ ]:


X_test_augmented.shape


# In[ ]:


np.min(X_train[3]), np.max(X_train[3]), np.mean(X_train[3]), np.std(X_train[3])


# In[ ]:


np.min(X_test_augmented[5]), np.max(X_test_augmented[5]), np.mean(X_test_augmented[5]), np.std(X_test_augmented[5])


# In[ ]:


plt.figure(figsize=(28,28))
test_image = X_test_augmented[50]
new_image = np.minimum(test_image, 255)
new_image[new_image < 50] = 0
plt.subplot(121),plt.imshow(test_image , cmap='gray')
plt.subplot(122),plt.imshow(new_image , cmap='gray')
plt.show()


# In[ ]:


images = get_samples(40, X_test_augmented)
plot_images(images)


# Note: pay close attention to this test set. This test set is slightly different from the train set. In order to improve your model, make sure you know what the difference is so that you can perform appropriate processings.
# 
# ** From my observation, the images in the test set are not centered and kind of shifting around a few pixels **

# # Submit your predictions as csv file

# In[ ]:


# let's make a silly prediction that every image is T-shirt, meaning every prediction is 0
# Here is how you can make such prediction
predictions = np.zeros(shape=[len(X_test_augmented),]).astype(int)


# In[ ]:


predictions.shape # make sure that you have 40000 predictions, since the hidden test set has 40000 images


# In[ ]:


pred_df = pd.DataFrame(predictions,columns=['pred'])
pred_df.head()


# In[ ]:


#MY_NAME='quantran'
#
pred_df.to_csv(PATH + f"/submissions/{MY_NAME}_submission.csv", index=None)


# By running the cell above, you actually submit your predictions directly to the submissions folder in Weekly_Project folder, as I have granted you permission to save files there. Let me know if you have any problem running the cell above.
# 
# 
# 
# Good luck!

# In[ ]:


import cv2 as cv


# In[ ]:


from matplotlib import pyplot as plt


# In[ ]:


img = cv.imread('test1.jpg')
print(type(img))
dst = cv.fastNlMeansDenoisingColored(img,None,10,10,7,21)
plt.subplot(121),plt.imshow(img)
plt.subplot(122),plt.imshow(dst)
plt.show()


# In[ ]:


img.shape


# In[ ]:




