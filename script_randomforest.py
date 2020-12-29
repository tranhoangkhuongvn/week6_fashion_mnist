
import numpy as np
import pandas as pd
import seaborn as sns
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift

import pprint
pp = pprint.PrettyPrinter(indent=4)



from tqdm import tqdm


from tensorflow.keras.datasets import fashion_mnist



#### Helper Functions ####

def shift_pixels(src, x, y):
    """
    shift x pixels vertically and y pixels horizontally
    """
    output = shift(src, [x,y], cval=0)
    
    return output





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


def get_samples(n_samples, X, y=None):
    '''Get n_samples randomly'''
    samples_index = np.random.choice(np.arange(len(X)), n_samples, replace=False)
    if not y is None:
        return X[samples_index], y[samples_index]
    return X[samples_index]








# #### Pre-process data:
# - Scale pixel values
# - Augment the data: shift the images, denoise the images
# - Flatten the 2D array --> 1D vector
# 

# 


from sklearn.base import BaseEstimator, TransformerMixin

class DenoiseTransformer(BaseEstimator, TransformerMixin):
	def __init__(self):
		pass

	def fit(self, X, y=None):
		return self
  	
	def denoise(self, src_img, threshold=50, display=False):
    
		new_image = np.minimum(src_img, 255)
		new_image[new_image < threshold] = 0
		if display:
			plt.figure(figsize=(15,15))
			plt.subplot(121),plt.imshow(src_img , cmap='gray')
			plt.subplot(122),plt.imshow(new_image , cmap='gray')
		return new_image  


	def transform(self, X, y=None):
		#return np.array([skimage.color.rgb2gray(img) for img in X])
		return np.array([self.denoise(img) for img in X])



class ShiftTransformer(BaseEstimator, TransformerMixin):
	def __init__(self, y):
		self.y = y
		self.n = y.shape[0]
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
		X_out = np.zeros((self.n * 9, 28, 28))
		y_out = np.zeros((self.n * 9, ))
		for i in tqdm(range(X.shape[0])):
			X_out[i*9: i*9+9], y_out[i*9:i*9+9] = self.image_generator(X[i], self.y[i])
			
		return X_out, y_out



(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()


print('Training data:', X_train.shape, y_train.shape)
print('Test data:', X_test.shape, y_test.shape)


denoise_train = DenoiseTransformer()
shift_train = ShiftTransformer(y_train)

denoise_test = DenoiseTransformer()
shift_test = ShiftTransformer(y_test)

denoised_X_train = denoise_train.fit_transform(X_train)
new_X_train, new_y_train = shift_train.fit_transform(denoised_X_train) 

denoised_X_test = denoise_test.fit_transform(X_test)
new_X_test, new_y_test = shift_test.fit_transform(denoised_X_test)

print('New training data: ', new_X_train.shape)
print('New training label:', new_y_train.shape)

print('New test data: ', new_X_test.shape)
print('New test label:', new_y_test.shape)

# Normalization
X_train_flat = new_X_train.reshape((-1, 784)) / 255
X_test_flat = new_X_test.reshape((-1, 784))/255

print('Training data after normalize:', X_train_flat.shape)
print('Test data after normalize:', X_test_flat.shape)


from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier





print('Random Forest Training....')

model_RF = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_leaf=8)
model_RF.fit(X_train_flat, new_y_train)


y_train_pred = model_RF.predict(X_train_flat)
y_test_pred = model_RF.predict(X_test_flat)
print('RF Train accuracy: ', accuracy_score(new_y_train, y_train_pred))
print("RF Test accuracy:", accuracy_score(new_y_test, y_test_pred))


import pickle

pickle.dump(model_RF, open('random_forest.pkl', 'wb'))

exit(0)















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




