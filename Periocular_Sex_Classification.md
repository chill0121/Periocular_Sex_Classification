# Sex Classification using Low-Res Periocular Images

This project focuses on using an image dataset of eyes, which includes not only the iris but also the surrounding periocular region, to build Deep Convolutional Neural Network (DCNN) classifiers for identifying the sex of the individual in the image. A key aspect of the project is analyzing the image features extracted by the models to identify those most relevant for sex classification. Additionally, multiple DCNN architectures will be explored, with models trained on two versions of the same dataset: one with the original images and another with randomly augmented images. The training and performance of these models will be compared to evaluate the impact of data augmentation.

Identifying the sex of an individual from periocular images has important applications in various fields, from security and biometric authentication to medical diagnostics. In security, such models can enhance identity verification systems in situations where only partial or low-resolution facial data is available. Moreover, automated sex classification using periocular regions could improve the accuracy of systems used in demographics analysis and research. By building classifiers that focus on this subtle but ubiquitous piece of an individual's biometrics, this project addresses the need for more adaptable, hopefully privacy-conscious, and efficient systems in multiple real-world scenarios with limited or non-ideal data.

https://github.com/chill0121/Periocular_Sex_Classification

---

*Note on the word-choice in this project:*

*The words "sex" and "gender" are often incorrectly used interchangeably in resources on this subject and broadly within society. Gender refers to characteristics largely assigned by cultural norms while sex are biological and physiologic characteristics. It has been decided that within this project the target class label used will be sex. However, depending on the efficacy of the model and image characteristics it uses to classify eyes, a strong argument could be made that the models are indeed classifying gender instead of sex, and I believe it is important to explore this distinction as it helps define what the model is actually identifying -- more on this in the Model Analysis and Feature Extraction (Section 8) and Conclusion (Section 10).*

## Table of Contents <a name="toc"></a>

---

- 1.[**Data Source Information**](#datasource)
- 2.[**Setup**](#setup)
  - 2.1. [Environment Details for Reproducility](#env)
  - 2.2. [Importing the Data](#dataimport)
- 3.[**Data Preprocessing**](#datapre)
  - 3.1. [First Looks](#firstlook)
  - 3.2. [Class Labels](#class)
  - 3.3. [Balance Dataset](#balance)
  - 3.4. [Image Normalization](#norm)
- 4.[**Train / Val / Test Split**](#split)
- 5.[**Exploratory Data Analysis (EDA)**](#eda)
- 6.[**Image Transformations / Augmentations**](#transforms)
- 7.[**Models**](#models)
  - 7.0. [Model Helper Functions](#helper)
  - 7.1. [Baseline Models](#baseline)
  - 7.2. [Deep Learning Models](#deep)
    - 7.2.1. [Shallow Feedforward Neural Network (FNN)](#fnn)
    - 7.2.2. [Deep-ish Convolutional Neural Network (CNN)](#cnn)
    - 7.2.3. [Deep Convolutional Neural Network (DCNN)](#dcnn)
- 8.[**Model Analysis and Feature Extraction Discussion**](#analysis)
  - 8.1. [Visualize Model Filters](#filters)
  - 8.2. [Occlusion Sensitivity Plots](#occlusion)
  - 8.3. [Principal Component Analysis (PCA) of Feature Embeddings](#pca)
  - 8.4. [Misclassification Exploration](#misclass)
  - 8.5. [Predict for Fun](#fun)
- 9.[**Results**](#results)
- 10.[**Conclusion**](#conclusion)
  - 10.1. [Limitations](#limitations)
  - 10.2. [Future Work](#future)

- [**Appendix A - Online References**](#appendixa)

## 1. Data Source Information <a name="datasource"></a>

---


**Source:** https://www.kaggle.com/datasets/pavelbiz/eyes-rtte/data

**Description:**

By using OpenCV Haar Eye Cascade (https://github.com/anaustinbeing/haar-cascade-files/blob/master/haarcascade_eye.xml), by the Kaggle user pavelbiz, scraped eye images from the website https://ruskino.ru which is a IMDB-type website which houses actor, director, writer, etc information.

**Contents:**
- RGB Images of individual eyes and surrounding periocular region at various sizes.
- 5,203 Female
- 6,324 Male

*Dataset size eventually parsed down to 5182 images at 56 x 56 x 3 pixels (2591 Female, 2591 Male). Reason and method is described in section 2.2.*

###### [Back to Table of Contents](#toc)

## 2. Setup <a name="setup"></a>

---


```python
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patheffects as path_effects
from PIL import Image

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, auc, roc_curve
from sklearn.decomposition import PCA

import tensorflow as tf
from tf_explain.core.occlusion_sensitivity import OcclusionSensitivity
import visualkeras
```

### 2.1. Environment Information for Reproducibility: <a name="env"></a>


```python
print(f"Python version: {sys.version}")

packages = [pd, np, sns, sklearn, tf]
for package in packages:
    print(f"{str(package).partition('from')[0]} using version: {package.__version__}")
```

    Python version: 3.11.9 (main, Apr  2 2024, 08:25:04) [Clang 15.0.0 (clang-1500.3.9.4)]
    <module 'pandas'  using version: 2.1.4
    <module 'numpy'  using version: 1.26.4
    <module 'seaborn'  using version: 0.13.2
    <module 'sklearn'  using version: 1.3.2
    <module 'tensorflow'  using version: 2.16.2


###### [Back to Table of Contents](#toc)

### 2.2. Importing the Data: <a name="dataimport"></a>

*Note: Two dataset characteristics were noticed during initial data import.*

1. Image sizes varied.
2. Numerically consecutive images showed left then right eyes of one individual. While it seems most images follow this pattern, some individuals only have one eye present where a number is then skipped. Unfortunately the pattern does not persist perfectly and it was noted that some individuals left and right eyes were not consecutively named (e.g. `68` ... `71`). This presents issues for splitting training and testing sets.

The fixes chosen to overcome these two issues:

1. The mean and median size of all images was shown to be ~ 56 x 56 pixels.
    - All images are to be resized to 56 x 56 and antialiased/resampled using `LANCZOS`.
2. Since naming conventions were not perfect in the creation of this dataset, it was decided to only use odd numbered images for the best chances at avoiding the same individual showing up in the dataset twice - mitigating data leakage between training and test sets. (An alternative could be to manually choose to split the images at a numerical cutoff ensuring the cutoff isn't between the same individual.)
    - Unfortunately this significantly reduces the dataset size, but this will be addressed and become a main focus of this project in the Section 6: Image Transformations/Augmentation.


```python
def image_import(file_list, sex, image_size):
    '''
    Takes a list of image filenames and loads the images into a dictionary with the same filename(*).

    (*) Adds up to 4 leading zeros to the name if they are not present.
    
    Parameters:
        file_list: List of filenames in the form ['name.extension', ...]
        sex: String denoting the folder in ./data to find the images. ('Male' or 'Female')
        image_size: Integer, will all images resize to this parameter. 
            All images are different sizes, median/mean of dataset is 56x56.
    Returns:
        eyes_dict: Dictionary whose keys are the filename with leading zeros and value is a numpy array of the image.
    '''
    eyes_dict = {}
    # img_sizes = []
    for file in file_list:
        # Separate file extension and name.
        number, _ = file.split('.')
        number_new = number.zfill(4) # Add leading zeros.
        # Only take odd numbers to remove more than 1 eye per person.
        if int(number) % 2 == 1:
            img = Image.open(f'./Data/{sex}/{number}.jpg')
            # Used to check most common size.
            #img_sizes.append(np.asarray(img).shape[0])
            
            img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
            eyes_dict[number_new] = np.asarray(img).flatten()

    return eyes_dict #, image_sizes
```


```python
img_size = 56 # Used throughout notebook.

female_files = os.listdir('./Data/Female')
male_files = os.listdir('./Data/male')

eyes_female = image_import(female_files, 'Female', image_size=img_size)
eyes_male = image_import(male_files, 'Male', image_size=img_size)
```

###### [Back to Table of Contents](#toc)

## 3. Data Preprocessing <a name="datapre"></a>

---

### 3.1. First Looks: <a name="firstlook"></a>

Let's take a look at a random image first and make sure it's the right size and expected format.


```python
print(eyes_female['0011'].reshape(img_size,img_size,3).shape)
plt.imshow(Image.fromarray(eyes_female['0011'].reshape(img_size,img_size,3)))
plt.show()
```

    (56, 56, 3)



    
![png](Periocular_Sex_Classification_files/Periocular_Sex_Classification_15_1.png)
    


Looks good - the shape of `(56, 56, 3)` indicates an 3-dimensional array representing RGB with no alpha channel.

###### [Back to Table of Contents](#toc)

### 3.2. Class Labels: <a name="class"></a>

Now, we should add the class labels to each image. This was made easier by keeping the two genders separate for now.


```python
eyes_female_df = pd.DataFrame(eyes_female.items(), columns = ['eye_d', 'rgb_flat'])
eyes_male_df = pd.DataFrame(eyes_male.items(), columns = ['eye_d', 'rgb_flat'])

eyes_female_df['sex'] = 0
eyes_male_df['sex'] = 1
```

###### [Back to Table of Contents](#toc)

### 3.3. Balance Dataset: <a name="balance"></a>

Now, we can see if the target variable is balanced or not.


```python
print(f'Female Sample Size: {len(eyes_female_df)}')
print(f'Male Sample Size: {len(eyes_male_df)}')
```

    Female Sample Size: 2591
    Male Sample Size: 3173


This shows a fairly unbalanced dataset which is important to keep in mind for when we do the train / val / test set splits and model evaluations.

In fact, to simplify things, we can balance the dataset by removing the extra male images by randomly sampling the male eyes without replacement to match the number of female eyes. Then, the two dataframes will be combined into one.


```python
# Balance classes, limiting the number of male eyes.
eyes_male_df = eyes_male_df.sample(n = len(eyes_female_df), random_state = 11, replace = False)

# Create dataframe with mixed sexes.
eyes_df = pd.concat([eyes_female_df, eyes_male_df]).reset_index(drop = True)
eyes_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>eye_d</th>
      <th>rgb_flat</th>
      <th>sex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0823</td>
      <td>[216, 170, 136, 214, 168, 134, 212, 166, 133, ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4217</td>
      <td>[141, 77, 49, 140, 76, 48, 138, 77, 48, 137, 7...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5109</td>
      <td>[244, 215, 199, 235, 206, 190, 165, 133, 118, ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4571</td>
      <td>[237, 210, 193, 234, 207, 190, 234, 205, 189, ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1409</td>
      <td>[175, 143, 130, 173, 141, 128, 172, 139, 130, ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5177</th>
      <td>0579</td>
      <td>[151, 112, 104, 149, 110, 103, 153, 114, 109, ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5178</th>
      <td>5191</td>
      <td>[210, 210, 210, 212, 212, 212, 209, 209, 209, ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5179</th>
      <td>0425</td>
      <td>[163, 118, 87, 164, 119, 88, 170, 125, 94, 166...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5180</th>
      <td>5491</td>
      <td>[66, 44, 47, 48, 26, 30, 46, 25, 31, 44, 25, 3...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5181</th>
      <td>0821</td>
      <td>[221, 159, 138, 221, 158, 138, 217, 150, 130, ...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5182 rows × 3 columns</p>
</div>



Now `eyes_df` is the combined balanced Male and Female dataset.

###### [Back to Table of Contents](#toc)

### 3.4. Image Normalization: <a name="norm"></a>

It's important to normalize the image pixel intensities for use in the neural networks.


```python
eyes_df['norm_flat'] = eyes_df['rgb_flat'] / 255
```


```python
# Function to reshape arrays for processes / models that require image array.
def flat_to_array(flat_array):
    '''
    Takes a dataframe column (series) and reshapes flat array to image array (H x W x C).
    '''
    return flat_array.reshape(img_size,img_size,3)
```

###### [Back to Table of Contents](#toc)

## 4. Train / Val / Test Split <a name="split"></a>

---

To effectively train the models we need to create a validation set to evaluate training on, and of course we need a holdout dataset to do the final evaluations once all model tweaking has been finalized. The class balance will be maintained between all subsets by using the `stratify` parameter in `train_test_split()`.

**Split Ratios:**

- Train = 70%
- Validation = 18%
- Test = 12%


```python
# Train and Test
X_train, X_test = train_test_split(eyes_df, test_size = 0.30, shuffle = True, stratify = eyes_df.sex, random_state = 11)
# Validation from Test
X_test, X_val = train_test_split(X_test, test_size = 0.60, shuffle = True, stratify = X_test.sex, random_state = 11)

# Create y.
y_train = X_train.sex
y_val = X_val.sex
y_test = X_test.sex

# Drop y from X.
X_train = X_train.drop(columns = ['sex'])
X_val = X_val.drop(columns = ['sex'])
X_test = X_test.drop(columns = ['sex'])

print(f'Train Set Size: {len(X_train)}')
print(f'Validation Set Size: {len(X_val)}')
print(f'Test Set Size: {len(X_test)}')
print('\n##################\n')
print('Target Class Balance:')
print('----Train:\n', y_train.value_counts())
print('----Validation:\n', y_val.value_counts())
print('----Test:\n', y_test.value_counts())
```

    Train Set Size: 3627
    Validation Set Size: 933
    Test Set Size: 622
    
    ##################
    
    Target Class Balance:
    ----Train:
     sex
    0    1814
    1    1813
    Name: count, dtype: int64
    ----Validation:
     sex
    1    467
    0    466
    Name: count, dtype: int64
    ----Test:
     sex
    0    311
    1    311
    Name: count, dtype: int64


###### [Back to Table of Contents](#toc)

## 5. Exploratory Data Analysis (EDA) <a name="eda"></a>

---

To avoid any data leakage, only images from the training set will be analyzed here.


```python
# Colors for the target classes to stay consistent.
cmap_sex = {'Female' :'#439A86', 'Male': '#423e80'}
```

To get a better idea of the brightness/intensity of the images within the dataset we can take the mean pixel value across the whole image and plot that distribution.


```python
# Find the mean pixel intensity of each image (down to one value along all axes).
mean_intensity = X_train.rgb_flat.apply(flat_to_array).apply(np.mean)

sns.histplot(mean_intensity, binwidth = 5, kde = True)
plt.xlabel('Mean Pixel Intensity Value')
plt.title('Pixel Intensity Histogram')
plt.show()
```


    
![png](Periocular_Sex_Classification_files/Periocular_Sex_Classification_35_0.png)
    


We can see that the distribution is centered around a pixel intensity of ~127 which means that the majority of images are not too light or dark. If we saw it positioned closer to the right, that means the photos would be darker and left, lighter. 

We do however, see that some majority light and dark photos do exist with this dataset represented by the left and right tails respectively. We will keep this in mind when deciding which random image augmentations to apply during training.

Next, let's separate the color channels and plot their distributions.


```python
mean_intensity_rgb = np.stack(X_train.rgb_flat.apply(flat_to_array).apply(np.mean, axis = 1).apply(np.mean, axis = 0).to_numpy())

# Plot the distribution of the mean intensities of individual RGB channels.
rgb_dict = {0:'Red', 1:'Green', 2:'Blue'}
for i in rgb_dict.keys():
    sns.histplot(mean_intensity_rgb[:,i], color = rgb_dict[i], binwidth = 5, kde = True)#, ax = ax[i])
plt.title('Mean Pixel Intensity of Each Channel (RGB)')
plt.legend(list(rgb_dict.values()))
plt.show()
```


    
![png](Periocular_Sex_Classification_files/Periocular_Sex_Classification_37_0.png)
    


The red distribution being further right indicates that warmer-tones are more common in the dataset. This also might mean that the majority of the dataset is caucasian, which makes sense based on the data source.

It might be interesting to see these RGB channel distributions separated by target class (sex).


```python
# Create a mask of the target class.
mask_sex = y_train == 0

# Plot the distribution of the mean intensities of individual channels in both Males and Females.
fig, ax = plt.subplots(1, 3, figsize = (14, 5), sharey = True)
for i in range(3):
    # Female
    sns.histplot(mean_intensity_rgb[:,i][mask_sex], kde = True, color = cmap_sex['Female'], ax = ax[i])
    # Male
    sns.histplot(mean_intensity_rgb[:,i][~mask_sex], kde = True, color = cmap_sex['Male'], ax = ax[i])
    ax[i].set_title(f'{rgb_dict[i]}')
fig.supxlabel('Mean Pixel Intensity')
fig.suptitle('Color Channel Distributions by Class Label')
plt.legend(['Female', 'Male'])
plt.show()
```


    
![png](Periocular_Sex_Classification_files/Periocular_Sex_Classification_39_0.png)
    


There doesn't seem to be much difference in distributions between the two classes, but it's always important to check.

###### [Back to Table of Contents](#toc)

## 6. Image Transformations / Augmentations <a name="transforms"></a>

---

While we did import the images into memory before, utilizing a dataloader and applying random image augmentations at each batch call should increase performance of the models. These transformations will only be performed on the training set and will introduce much more diversity than the original dataset.

The following augmentations will be performed randomly:
- Random Flip in the horizontal direction. Vertical isn't necessary here as it's rare that these models will see an eye in that orientation.
- Random Rotation. This will be very important to help the models become rotation invariant by helping them generalize much better than the original dataset which likely has an uneven distribution of eye angles.
- Random Crop. Will select a random crop of the image and interpolate the pixels to maintain the same image size. This will help account for different eye positions.
- Random Brightness. Again, this is helpful because the original dataset has varying light qualities.
- Random Saturation. Same as brightness, but with color saturation.


```python
BATCH_SIZE = 16
```


```python
# Create a dataset with images and labels together.
# Train.
X_ds_train = tf.data.Dataset.from_tensor_slices(np.stack(X_train.norm_flat.values).reshape(len(X_train), img_size, img_size, 3))
y_ds_train = tf.data.Dataset.from_tensor_slices(y_train.values)

ds_train = tf.data.Dataset.zip((X_ds_train, y_ds_train))

# Val.
X_ds_val = tf.data.Dataset.from_tensor_slices(np.stack(X_val.norm_flat.values).reshape(len(X_val), img_size, img_size, 3))
y_ds_val = tf.data.Dataset.from_tensor_slices(y_val.values)

ds_val = tf.data.Dataset.zip((X_ds_val, y_ds_val))
```


```python
# Transformation object.
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomCrop(img_size, img_size),
    tf.keras.layers.RandomBrightness(0.4, value_range = (0, 1)),
    # Random saturation using lambda.
    tf.keras.layers.Lambda(lambda x: tf.image.random_saturation(x, lower = 0, upper = 3))
    ])

# Autotune to set buffer size.
AUTOTUNE = tf.data.AUTOTUNE

# Function to setup the transformations within the dataloader.
def prepare(ds, shuffle = False, augment = False):
    if shuffle:
        ds = ds.shuffle(1000)

    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training = True), y), num_parallel_calls = AUTOTUNE)
    
  # Prefetch buffer necessary here to ensure proper loading during training.
    return ds.prefetch(buffer_size = AUTOTUNE)
```


```python
# Training Set Final
ds_train_transformed = prepare(ds_train, shuffle = True, augment = True)
ds_train_transformed = ds_train_transformed.batch(BATCH_SIZE)

# Validation Set Final
ds_val = prepare(ds_val, shuffle = False, augment = False)
ds_val = ds_val.batch(BATCH_SIZE)

# For comparison to transformed training.
ds_train = prepare(ds_train, shuffle = True, augment = False)
ds_train = ds_train.batch(BATCH_SIZE)
```


```python
for img, label in ds_train:
    fig, ax = plt.subplots(2, 3, sharex = True, sharey = True)
    img = (img.numpy() * 255).astype(np.uint8)
    for i in range(6):
        ax[i // 3, i % 3].imshow(img[i])
    fig.suptitle('Non-Augmented Training Images')
    break
```


    
![png](Periocular_Sex_Classification_files/Periocular_Sex_Classification_46_0.png)
    



```python
for img, label in ds_train_transformed:
    fig, ax = plt.subplots(2, 3, sharex = True, sharey = True)
    img = (img.numpy() * 255).astype(np.uint8)
    for i in range(6):
        ax[i // 3, i % 3].imshow(img[i])
    fig.suptitle('Augmented Training Images')
    break
```


    
![png](Periocular_Sex_Classification_files/Periocular_Sex_Classification_47_0.png)
    


###### [Back to Table of Contents](#toc)

## 7. Models <a name="models"></a>

---

As was mentioned above, since the dataset has been class balanced, the metric we will evaluate all models is accuracy. Of course, the models will be trained using the training set and will utilize the validation set as the evaluation set during training. Validation set accuracy scores will be output after model training, and the final test set scores will be calculated in the Results section (9).

Also, each model will be trained on the training set WITH image augmentation, but also trained on the original dataset with the same dataloader settings, just without augmentations, so we can compare the performance of including these transformations at the end.

### 7.0. Model Helper Functions: <a name="helper"></a>

Here we will add a few functions to help visualize how the model perform during training.


```python
def set_region_overlay(model_history_df, x_offset):
    x_mid = ((model_history_df.index.stop-1) + model_history_df.val_loss.idxmin()) / 2
    plt.text(x = x_mid - x_offset,
             y = (plt.ylim()[0] + plt.ylim()[1]) / 2,
             s = 'Early Stop',
             rotation = 'horizontal',
             weight = 'extra bold',
             fontsize = 'large',
             antialiased = True,
             alpha = 1,
             c = 'white',
             bbox = dict(facecolor = 'black', edgecolor = 'black', boxstyle = 'round', alpha = 0.5))
    return None

def plot_TF_training_history(model_history_df, plot_title = None):

    # Find all epochs that callback ReduceLROnPlateau() occurred.
    lr_change = model_history_df.learning_rate.shift(-1) != model_history_df.learning_rate

    # Create color map and lines style map for train/val
    plot_maps = {'cmap': {'accuracy': '#653096',
                        'loss': '#653096',
                        'val_accuracy': '#004a54',
                        'val_loss': '#004a54'},
                'dashmap': {'accuracy': '',
                            'loss': (2,1),
                            'val_accuracy': '',
                            'val_loss': (2,1)}}

    # Plot
    fig, ax = plt.subplots(figsize = (10,6))
    ax = sns.lineplot(model_history_df.drop(columns = ['learning_rate']).iloc[1:], palette = plot_maps['cmap'], dashes = plot_maps['dashmap'])
    ax.set_xlabel('Epoch')

    # Create secondary x-axis for Learning Rate changes.
    sec_ax = ax.secondary_xaxis('top')
    sec_ax.set_xticks(model_history_df[lr_change].index[:-1])
    sec_ax.set_xticklabels([f'{x:.1e}' for x in model_history_df[lr_change].learning_rate[1:]])
    sec_ax.tick_params(axis = 'x', which = 'major', labelsize = 7)
    sec_ax.set_xlabel('Learning Rate Reductions')

    # Create vertical line for each LR change.
    for epoch in (model_history_df[lr_change].index[:-1]):
        plt.axvline(x = epoch, c = '#d439ad', ls = (0, (5,5)))
    # Create lines for best epoch/val_loss.
    plt.axvline(x = (model_history_df.val_loss.idxmin()), c = '#f54260', ls = (0, (3,1,1,1)))
    plt.axhline(y = (model_history_df.val_loss.min()), c = '#f54260', alpha = 0.3, ls = (0, (3,1,1,1)))
    # Grey out epochs after early stop.
    plt.axvspan(model_history_df.val_loss.idxmin(), model_history_df.index.stop-1, facecolor = 'black', alpha = 0.25)
    plt.margins(x = 0)
    set_region_overlay(model_history_df, 5)

    if plot_title is not None:
        fig.suptitle(plot_title)

    plt.legend(loc = 'center')
    plt.show()
    return None
```

###### [Back to Table of Contents](#toc)

### 7.1. Baseline Models: <a name="baseline"></a>

It's always important to formulate a baseline model to compare all models to.

**Random Chance Baseline**

Only two classes here, so the random chance baseline is 50%.


```python
1/2
```




    0.5



**K-Nearest Neighbors (KNN) as Baseline**

Since this dataset contains all eyes that are mostly centered in the same position, KNN is a quick and easy model to implement and compare more complex methods against.


```python
# TODO: Implement KNN discriminant plot from sklearn.
mod_knn = KNeighborsClassifier(n_neighbors = 2, weights = 'distance', algorithm = 'brute', p = 2)
mod_knn.fit(np.stack(X_train.norm_flat), y_train)

y_pred_val_knn = mod_knn.predict(np.stack(X_val.norm_flat))
accuracy_score(y_val, y_pred_val_knn)
```




    0.7738478027867095



77.38% accuracy on the validation set is pretty impressive for a model only using distance metrics on pixel intensities. This surprise performance is likely because most eyes in this dataset are in relatively the same position.

Now the prediction on the test set will be performed and saved for the results section after all tuning is completed.


```python
# Predict on the test set.
y_pred_knn_proba = mod_knn.predict_proba(np.stack(X_test.norm_flat))
y_pred_knn = mod_knn.predict(np.stack(X_test.norm_flat))
```

###### [Back to Table of Contents](#toc)

### 7.2. Deep Learning Models: <a name="deep"></a>

#### 7.2.1. Shallow Feedforward Neural Network (FNN): <a name="fnn"></a>

Given that there are only two classes and one type of image (photos of eyes), it may be overlooked that a simple shallow dense network can potentially perform quite well on this classification task (see MNIST numbers dataset). While, it likely won't outperform a Convolutional Neural Network (CNN), it may be worth it to train what we'll call an FNN here to compare it to the CNN-based models.

We'll use two layers of dense nodes with ReLU activation functions and a light dropout (to mitigate overfitting) applied after activation of each of those two layers.


```python
def build_fnn(input_shape = (img_size, img_size, 3)):
    mod_fnn = tf.keras.Sequential([
        tf.keras.Input(shape = input_shape, name = 'Imput Image'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(128),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(2, name = 'Predictions')])
    return mod_fnn

mod_fnn = build_fnn()
mod_fnn.summary()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential_1"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ flatten (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9408</span>)           │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)            │     <span style="color: #00af00; text-decoration-color: #00af00">4,817,408</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ re_lu (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)                    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)            │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)            │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)            │        <span style="color: #00af00; text-decoration-color: #00af00">65,664</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ re_lu_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)                  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)            │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)            │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Predictions (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>)              │           <span style="color: #00af00; text-decoration-color: #00af00">258</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">4,883,330</span> (18.63 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">4,883,330</span> (18.63 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



Model Specifications:

- Loss Function: Sparse Categorical using Cross-Entropy
- Optimizer: Adam
- Callbacks:
    - Early Stopping
    - Adjust Learning Rate on Loss Plateau

Below we can see the two "Callbacks" initialized. These are very important for how these models are being trained.

Details:
1. Early Stopping:
    - We need a way to train the model a sufficient amount of epochs, but as validation loss is measured and if this loss stops improving over, in this case, 25 epochs -- the model will stop training and revert weights to values at the best epoch (lowest validation loss).
2. Reduce Learning Rate on Plateau:
    - In need of a method to appropriately adjust how fast the model performs its gradient descent, this appropriately named callback does just that.
    - Patience is set at 5: As the validation loss reduces but eventually stalls (plateaus) for 5 consecutive epochs, the learning rate will be reduced by a factor of 50%. This reduction in learning rate then has a cooldown period where it won't reduce for 8 epochs.

Both of these callbacks and all of their parameters were used and iteratively adjusted during training of all subsequent models.


```python
loss_fx = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True) # Integer labels.
optimizer_param = 'adam'
val_freq = 1
n_epochs = 250
# Callbacks to use:
early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 25, verbose = 1, restore_best_weights = True)
reduce_lr_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', 
                                                         factor = 0.5,
                                                         patience = 5,
                                                         cooldown = 8,
                                                         min_lr = 0.0001,
                                                         verbose = 1)
```


```python
mod_fnn.compile(optimizer = optimizer_param, loss = loss_fx, metrics = ['accuracy'])

mod_fnn_hist = mod_fnn.fit(
    ds_train_transformed,
    batch_size = BATCH_SIZE,
    epochs = n_epochs,
    verbose = "auto",
    callbacks = [early_stop, reduce_lr_plateau],
    #validation_split = 0.0,
    validation_data = ds_val,
    shuffle = True,
    class_weight = None,
    sample_weight = None,
    initial_epoch = 0,
    steps_per_epoch = None,
    validation_steps = None,
    validation_batch_size = None,
    validation_freq = val_freq)
```

    Epoch 1/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.5278 - loss: 2.1671 - val_accuracy: 0.5927 - val_loss: 0.6472 - learning_rate: 0.0010
    Epoch 2/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 11ms/step - accuracy: 0.6062 - loss: 0.6833 - val_accuracy: 0.6056 - val_loss: 0.6615 - learning_rate: 0.0010
    Epoch 3/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.5961 - loss: 0.6758 - val_accuracy: 0.5756 - val_loss: 0.6795 - learning_rate: 0.0010
    Epoch 4/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.6171 - loss: 0.6577 - val_accuracy: 0.5005 - val_loss: 0.7131 - learning_rate: 0.0010
    Epoch 5/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 11ms/step - accuracy: 0.5448 - loss: 0.6892 - val_accuracy: 0.6742 - val_loss: 0.6361 - learning_rate: 0.0010
    Epoch 6/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.6135 - loss: 0.6547 - val_accuracy: 0.7085 - val_loss: 0.6061 - learning_rate: 0.0010
    Epoch 7/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.6528 - loss: 0.6352 - val_accuracy: 0.5273 - val_loss: 0.6927 - learning_rate: 0.0010
    Epoch 8/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.6448 - loss: 0.6366 - val_accuracy: 0.6956 - val_loss: 0.5839 - learning_rate: 0.0010
    Epoch 9/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.6817 - loss: 0.6157 - val_accuracy: 0.6506 - val_loss: 0.6462 - learning_rate: 0.0010
    Epoch 10/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.6851 - loss: 0.6095 - val_accuracy: 0.7556 - val_loss: 0.5567 - learning_rate: 0.0010
    Epoch 11/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.6576 - loss: 0.6245 - val_accuracy: 0.6699 - val_loss: 0.6260 - learning_rate: 0.0010
    Epoch 12/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.6724 - loss: 0.6188 - val_accuracy: 0.5348 - val_loss: 0.7265 - learning_rate: 0.0010
    Epoch 13/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.6414 - loss: 0.6337 - val_accuracy: 0.6924 - val_loss: 0.5991 - learning_rate: 0.0010
    Epoch 14/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.6734 - loss: 0.6304 - val_accuracy: 0.7192 - val_loss: 0.5567 - learning_rate: 0.0010
    Epoch 15/250
    [1m226/227[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 10ms/step - accuracy: 0.6744 - loss: 0.6125
    Epoch 15: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.6745 - loss: 0.6125 - val_accuracy: 0.7181 - val_loss: 0.5605 - learning_rate: 0.0010
    Epoch 16/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.6886 - loss: 0.5945 - val_accuracy: 0.7513 - val_loss: 0.5400 - learning_rate: 5.0000e-04
    Epoch 17/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 11ms/step - accuracy: 0.7160 - loss: 0.5832 - val_accuracy: 0.7556 - val_loss: 0.5297 - learning_rate: 5.0000e-04
    Epoch 18/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7145 - loss: 0.5865 - val_accuracy: 0.7621 - val_loss: 0.5279 - learning_rate: 5.0000e-04
    Epoch 19/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7130 - loss: 0.5730 - val_accuracy: 0.7074 - val_loss: 0.5564 - learning_rate: 5.0000e-04
    Epoch 20/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7051 - loss: 0.5819 - val_accuracy: 0.7460 - val_loss: 0.5305 - learning_rate: 5.0000e-04
    Epoch 21/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.6952 - loss: 0.5907 - val_accuracy: 0.7556 - val_loss: 0.5296 - learning_rate: 5.0000e-04
    Epoch 22/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7003 - loss: 0.5899 - val_accuracy: 0.6913 - val_loss: 0.5659 - learning_rate: 5.0000e-04
    Epoch 23/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.6935 - loss: 0.5938 - val_accuracy: 0.7331 - val_loss: 0.5399 - learning_rate: 5.0000e-04
    Epoch 24/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7053 - loss: 0.5879 - val_accuracy: 0.7578 - val_loss: 0.5221 - learning_rate: 5.0000e-04
    Epoch 25/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.6995 - loss: 0.5798 - val_accuracy: 0.7556 - val_loss: 0.5289 - learning_rate: 5.0000e-04
    Epoch 26/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.6949 - loss: 0.5883 - val_accuracy: 0.6259 - val_loss: 0.6715 - learning_rate: 5.0000e-04
    Epoch 27/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.6971 - loss: 0.5866 - val_accuracy: 0.7567 - val_loss: 0.5245 - learning_rate: 5.0000e-04
    Epoch 28/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7184 - loss: 0.5727 - val_accuracy: 0.7331 - val_loss: 0.5397 - learning_rate: 5.0000e-04
    Epoch 29/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 10ms/step - accuracy: 0.6896 - loss: 0.5951
    Epoch 29: ReduceLROnPlateau reducing learning rate to 0.0002500000118743628.
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.6897 - loss: 0.5950 - val_accuracy: 0.6860 - val_loss: 0.5609 - learning_rate: 5.0000e-04
    Epoch 30/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7223 - loss: 0.5695 - val_accuracy: 0.7256 - val_loss: 0.5346 - learning_rate: 2.5000e-04
    Epoch 31/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7203 - loss: 0.5684 - val_accuracy: 0.7588 - val_loss: 0.5087 - learning_rate: 2.5000e-04
    Epoch 32/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7314 - loss: 0.5630 - val_accuracy: 0.7653 - val_loss: 0.5117 - learning_rate: 2.5000e-04
    Epoch 33/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7172 - loss: 0.5780 - val_accuracy: 0.7642 - val_loss: 0.5322 - learning_rate: 2.5000e-04
    Epoch 34/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7122 - loss: 0.5770 - val_accuracy: 0.7674 - val_loss: 0.5251 - learning_rate: 2.5000e-04
    Epoch 35/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7180 - loss: 0.5665 - val_accuracy: 0.7642 - val_loss: 0.5087 - learning_rate: 2.5000e-04
    Epoch 36/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7288 - loss: 0.5608 - val_accuracy: 0.7556 - val_loss: 0.5166 - learning_rate: 2.5000e-04
    Epoch 37/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7148 - loss: 0.5716 - val_accuracy: 0.7578 - val_loss: 0.5150 - learning_rate: 2.5000e-04
    Epoch 38/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7197 - loss: 0.5587 - val_accuracy: 0.7653 - val_loss: 0.5113 - learning_rate: 2.5000e-04
    Epoch 39/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7154 - loss: 0.5623 - val_accuracy: 0.7203 - val_loss: 0.5411 - learning_rate: 2.5000e-04
    Epoch 40/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7195 - loss: 0.5670 - val_accuracy: 0.7481 - val_loss: 0.5167 - learning_rate: 2.5000e-04
    Epoch 41/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7432 - loss: 0.5531 - val_accuracy: 0.7760 - val_loss: 0.5075 - learning_rate: 2.5000e-04
    Epoch 42/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7202 - loss: 0.5734 - val_accuracy: 0.7803 - val_loss: 0.5057 - learning_rate: 2.5000e-04
    Epoch 43/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7083 - loss: 0.5624 - val_accuracy: 0.7610 - val_loss: 0.5035 - learning_rate: 2.5000e-04
    Epoch 44/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7067 - loss: 0.5574 - val_accuracy: 0.7792 - val_loss: 0.4961 - learning_rate: 2.5000e-04
    Epoch 45/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7352 - loss: 0.5504 - val_accuracy: 0.7546 - val_loss: 0.5094 - learning_rate: 2.5000e-04
    Epoch 46/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7206 - loss: 0.5625 - val_accuracy: 0.7717 - val_loss: 0.4922 - learning_rate: 2.5000e-04
    Epoch 47/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7306 - loss: 0.5493 - val_accuracy: 0.7878 - val_loss: 0.4931 - learning_rate: 2.5000e-04
    Epoch 48/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7370 - loss: 0.5420 - val_accuracy: 0.7792 - val_loss: 0.5021 - learning_rate: 2.5000e-04
    Epoch 49/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7432 - loss: 0.5403 - val_accuracy: 0.7728 - val_loss: 0.4857 - learning_rate: 2.5000e-04
    Epoch 50/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7394 - loss: 0.5457 - val_accuracy: 0.7856 - val_loss: 0.5032 - learning_rate: 2.5000e-04
    Epoch 51/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7357 - loss: 0.5541 - val_accuracy: 0.7749 - val_loss: 0.4863 - learning_rate: 2.5000e-04
    Epoch 52/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7441 - loss: 0.5485 - val_accuracy: 0.7406 - val_loss: 0.5219 - learning_rate: 2.5000e-04
    Epoch 53/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7263 - loss: 0.5487 - val_accuracy: 0.7867 - val_loss: 0.4823 - learning_rate: 2.5000e-04
    Epoch 54/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7442 - loss: 0.5336 - val_accuracy: 0.7792 - val_loss: 0.4807 - learning_rate: 2.5000e-04
    Epoch 55/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7309 - loss: 0.5418 - val_accuracy: 0.7567 - val_loss: 0.5111 - learning_rate: 2.5000e-04
    Epoch 56/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7344 - loss: 0.5410 - val_accuracy: 0.7696 - val_loss: 0.5007 - learning_rate: 2.5000e-04
    Epoch 57/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7358 - loss: 0.5387 - val_accuracy: 0.7867 - val_loss: 0.4766 - learning_rate: 2.5000e-04
    Epoch 58/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7374 - loss: 0.5282 - val_accuracy: 0.7535 - val_loss: 0.5061 - learning_rate: 2.5000e-04
    Epoch 59/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7340 - loss: 0.5403 - val_accuracy: 0.7803 - val_loss: 0.4748 - learning_rate: 2.5000e-04
    Epoch 60/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7272 - loss: 0.5568 - val_accuracy: 0.7310 - val_loss: 0.5188 - learning_rate: 2.5000e-04
    Epoch 61/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7365 - loss: 0.5432 - val_accuracy: 0.7460 - val_loss: 0.5156 - learning_rate: 2.5000e-04
    Epoch 62/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7228 - loss: 0.5544 - val_accuracy: 0.7792 - val_loss: 0.4784 - learning_rate: 2.5000e-04
    Epoch 63/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7327 - loss: 0.5305 - val_accuracy: 0.7706 - val_loss: 0.4859 - learning_rate: 2.5000e-04
    Epoch 64/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 10ms/step - accuracy: 0.7575 - loss: 0.5259
    Epoch 64: ReduceLROnPlateau reducing learning rate to 0.0001250000059371814.
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7575 - loss: 0.5259 - val_accuracy: 0.7856 - val_loss: 0.4804 - learning_rate: 2.5000e-04
    Epoch 65/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7532 - loss: 0.5235 - val_accuracy: 0.7953 - val_loss: 0.4709 - learning_rate: 1.2500e-04
    Epoch 66/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7425 - loss: 0.5269 - val_accuracy: 0.7867 - val_loss: 0.4709 - learning_rate: 1.2500e-04
    Epoch 67/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7514 - loss: 0.5182 - val_accuracy: 0.7803 - val_loss: 0.4715 - learning_rate: 1.2500e-04
    Epoch 68/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7430 - loss: 0.5236 - val_accuracy: 0.7942 - val_loss: 0.4691 - learning_rate: 1.2500e-04
    Epoch 69/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7476 - loss: 0.5306 - val_accuracy: 0.7878 - val_loss: 0.4687 - learning_rate: 1.2500e-04
    Epoch 70/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7573 - loss: 0.5227 - val_accuracy: 0.7942 - val_loss: 0.4684 - learning_rate: 1.2500e-04
    Epoch 71/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7423 - loss: 0.5359 - val_accuracy: 0.7878 - val_loss: 0.4710 - learning_rate: 1.2500e-04
    Epoch 72/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7535 - loss: 0.5218 - val_accuracy: 0.7974 - val_loss: 0.4651 - learning_rate: 1.2500e-04
    Epoch 73/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7548 - loss: 0.5164 - val_accuracy: 0.7899 - val_loss: 0.4698 - learning_rate: 1.2500e-04
    Epoch 74/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 11ms/step - accuracy: 0.7643 - loss: 0.5066 - val_accuracy: 0.7856 - val_loss: 0.4674 - learning_rate: 1.2500e-04
    Epoch 75/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7628 - loss: 0.4966 - val_accuracy: 0.7835 - val_loss: 0.4653 - learning_rate: 1.2500e-04
    Epoch 76/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7610 - loss: 0.5117 - val_accuracy: 0.7835 - val_loss: 0.4660 - learning_rate: 1.2500e-04
    Epoch 77/250
    [1m223/227[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 10ms/step - accuracy: 0.7449 - loss: 0.5319
    Epoch 77: ReduceLROnPlateau reducing learning rate to 0.0001.
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7450 - loss: 0.5317 - val_accuracy: 0.7996 - val_loss: 0.4651 - learning_rate: 1.2500e-04
    Epoch 78/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7680 - loss: 0.5106 - val_accuracy: 0.7899 - val_loss: 0.4664 - learning_rate: 1.0000e-04
    Epoch 79/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7484 - loss: 0.5119 - val_accuracy: 0.8017 - val_loss: 0.4626 - learning_rate: 1.0000e-04
    Epoch 80/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7512 - loss: 0.5095 - val_accuracy: 0.7942 - val_loss: 0.4614 - learning_rate: 1.0000e-04
    Epoch 81/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7394 - loss: 0.5417 - val_accuracy: 0.7556 - val_loss: 0.5083 - learning_rate: 1.0000e-04
    Epoch 82/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7530 - loss: 0.5253 - val_accuracy: 0.7985 - val_loss: 0.4602 - learning_rate: 1.0000e-04
    Epoch 83/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7509 - loss: 0.5223 - val_accuracy: 0.7878 - val_loss: 0.4686 - learning_rate: 1.0000e-04
    Epoch 84/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7497 - loss: 0.5185 - val_accuracy: 0.7974 - val_loss: 0.4611 - learning_rate: 1.0000e-04
    Epoch 85/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7741 - loss: 0.5092 - val_accuracy: 0.7996 - val_loss: 0.4605 - learning_rate: 1.0000e-04
    Epoch 86/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7439 - loss: 0.5128 - val_accuracy: 0.8049 - val_loss: 0.4647 - learning_rate: 1.0000e-04
    Epoch 87/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7612 - loss: 0.5120 - val_accuracy: 0.8049 - val_loss: 0.4596 - learning_rate: 1.0000e-04
    Epoch 88/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7510 - loss: 0.5080 - val_accuracy: 0.8006 - val_loss: 0.4585 - learning_rate: 1.0000e-04
    Epoch 89/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7573 - loss: 0.5123 - val_accuracy: 0.8060 - val_loss: 0.4556 - learning_rate: 1.0000e-04
    Epoch 90/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7578 - loss: 0.5032 - val_accuracy: 0.7867 - val_loss: 0.4603 - learning_rate: 1.0000e-04
    Epoch 91/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7661 - loss: 0.5007 - val_accuracy: 0.7867 - val_loss: 0.4624 - learning_rate: 1.0000e-04
    Epoch 92/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7408 - loss: 0.5173 - val_accuracy: 0.7974 - val_loss: 0.4602 - learning_rate: 1.0000e-04
    Epoch 93/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7578 - loss: 0.5027 - val_accuracy: 0.7771 - val_loss: 0.4704 - learning_rate: 1.0000e-04
    Epoch 94/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7685 - loss: 0.4967 - val_accuracy: 0.7964 - val_loss: 0.4549 - learning_rate: 1.0000e-04
    Epoch 95/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7637 - loss: 0.4983 - val_accuracy: 0.7867 - val_loss: 0.4587 - learning_rate: 1.0000e-04
    Epoch 96/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7720 - loss: 0.4878 - val_accuracy: 0.7899 - val_loss: 0.4583 - learning_rate: 1.0000e-04
    Epoch 97/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7608 - loss: 0.5024 - val_accuracy: 0.7996 - val_loss: 0.4570 - learning_rate: 1.0000e-04
    Epoch 98/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7560 - loss: 0.5131 - val_accuracy: 0.8017 - val_loss: 0.4663 - learning_rate: 1.0000e-04
    Epoch 99/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7682 - loss: 0.5004 - val_accuracy: 0.8006 - val_loss: 0.4563 - learning_rate: 1.0000e-04
    Epoch 100/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7788 - loss: 0.4876 - val_accuracy: 0.7931 - val_loss: 0.4519 - learning_rate: 1.0000e-04
    Epoch 101/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7668 - loss: 0.5059 - val_accuracy: 0.7996 - val_loss: 0.4624 - learning_rate: 1.0000e-04
    Epoch 102/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7507 - loss: 0.5218 - val_accuracy: 0.7696 - val_loss: 0.4712 - learning_rate: 1.0000e-04
    Epoch 103/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7647 - loss: 0.4958 - val_accuracy: 0.8049 - val_loss: 0.4533 - learning_rate: 1.0000e-04
    Epoch 104/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7688 - loss: 0.4858 - val_accuracy: 0.8039 - val_loss: 0.4532 - learning_rate: 1.0000e-04
    Epoch 105/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7608 - loss: 0.5013 - val_accuracy: 0.8039 - val_loss: 0.4507 - learning_rate: 1.0000e-04
    Epoch 106/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7580 - loss: 0.4954 - val_accuracy: 0.7985 - val_loss: 0.4538 - learning_rate: 1.0000e-04
    Epoch 107/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7628 - loss: 0.5039 - val_accuracy: 0.7996 - val_loss: 0.4555 - learning_rate: 1.0000e-04
    Epoch 108/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7569 - loss: 0.5036 - val_accuracy: 0.8124 - val_loss: 0.4520 - learning_rate: 1.0000e-04
    Epoch 109/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7563 - loss: 0.5058 - val_accuracy: 0.8071 - val_loss: 0.4521 - learning_rate: 1.0000e-04
    Epoch 110/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7701 - loss: 0.5028 - val_accuracy: 0.8114 - val_loss: 0.4577 - learning_rate: 1.0000e-04
    Epoch 111/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7793 - loss: 0.4833 - val_accuracy: 0.7996 - val_loss: 0.4529 - learning_rate: 1.0000e-04
    Epoch 112/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7634 - loss: 0.5023 - val_accuracy: 0.8039 - val_loss: 0.4466 - learning_rate: 1.0000e-04
    Epoch 113/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7796 - loss: 0.4849 - val_accuracy: 0.7910 - val_loss: 0.4517 - learning_rate: 1.0000e-04
    Epoch 114/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7768 - loss: 0.4866 - val_accuracy: 0.8071 - val_loss: 0.4528 - learning_rate: 1.0000e-04
    Epoch 115/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 11ms/step - accuracy: 0.7723 - loss: 0.4868 - val_accuracy: 0.7910 - val_loss: 0.4551 - learning_rate: 1.0000e-04
    Epoch 116/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 11ms/step - accuracy: 0.7594 - loss: 0.4990 - val_accuracy: 0.8049 - val_loss: 0.4488 - learning_rate: 1.0000e-04
    Epoch 117/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7585 - loss: 0.5068 - val_accuracy: 0.8071 - val_loss: 0.4499 - learning_rate: 1.0000e-04
    Epoch 118/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7502 - loss: 0.4991 - val_accuracy: 0.8017 - val_loss: 0.4495 - learning_rate: 1.0000e-04
    Epoch 119/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7747 - loss: 0.4887 - val_accuracy: 0.8081 - val_loss: 0.4473 - learning_rate: 1.0000e-04
    Epoch 120/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7760 - loss: 0.4903 - val_accuracy: 0.8071 - val_loss: 0.4567 - learning_rate: 1.0000e-04
    Epoch 121/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7530 - loss: 0.5066 - val_accuracy: 0.8017 - val_loss: 0.4453 - learning_rate: 1.0000e-04
    Epoch 122/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7699 - loss: 0.4794 - val_accuracy: 0.7921 - val_loss: 0.4526 - learning_rate: 1.0000e-04
    Epoch 123/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7645 - loss: 0.5037 - val_accuracy: 0.7856 - val_loss: 0.4560 - learning_rate: 1.0000e-04
    Epoch 124/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7570 - loss: 0.4973 - val_accuracy: 0.7964 - val_loss: 0.4543 - learning_rate: 1.0000e-04
    Epoch 125/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7652 - loss: 0.4843 - val_accuracy: 0.7867 - val_loss: 0.4642 - learning_rate: 1.0000e-04
    Epoch 126/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7527 - loss: 0.4993 - val_accuracy: 0.7964 - val_loss: 0.4457 - learning_rate: 1.0000e-04
    Epoch 127/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7753 - loss: 0.4791 - val_accuracy: 0.8049 - val_loss: 0.4412 - learning_rate: 1.0000e-04
    Epoch 128/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7745 - loss: 0.4897 - val_accuracy: 0.8028 - val_loss: 0.4493 - learning_rate: 1.0000e-04
    Epoch 129/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7479 - loss: 0.5164 - val_accuracy: 0.8028 - val_loss: 0.4448 - learning_rate: 1.0000e-04
    Epoch 130/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7654 - loss: 0.4893 - val_accuracy: 0.8146 - val_loss: 0.4498 - learning_rate: 1.0000e-04
    Epoch 131/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7679 - loss: 0.4846 - val_accuracy: 0.8039 - val_loss: 0.4575 - learning_rate: 1.0000e-04
    Epoch 132/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7634 - loss: 0.5007 - val_accuracy: 0.7910 - val_loss: 0.4551 - learning_rate: 1.0000e-04
    Epoch 133/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7667 - loss: 0.5045 - val_accuracy: 0.8017 - val_loss: 0.4552 - learning_rate: 1.0000e-04
    Epoch 134/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7610 - loss: 0.4902 - val_accuracy: 0.7728 - val_loss: 0.4647 - learning_rate: 1.0000e-04
    Epoch 135/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7531 - loss: 0.5082 - val_accuracy: 0.7760 - val_loss: 0.4710 - learning_rate: 1.0000e-04
    Epoch 136/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7501 - loss: 0.5070 - val_accuracy: 0.8103 - val_loss: 0.4427 - learning_rate: 1.0000e-04
    Epoch 137/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7751 - loss: 0.4925 - val_accuracy: 0.8039 - val_loss: 0.4390 - learning_rate: 1.0000e-04
    Epoch 138/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7701 - loss: 0.4965 - val_accuracy: 0.8017 - val_loss: 0.4426 - learning_rate: 1.0000e-04
    Epoch 139/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7551 - loss: 0.5084 - val_accuracy: 0.8049 - val_loss: 0.4449 - learning_rate: 1.0000e-04
    Epoch 140/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7639 - loss: 0.4881 - val_accuracy: 0.8135 - val_loss: 0.4471 - learning_rate: 1.0000e-04
    Epoch 141/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7726 - loss: 0.4913 - val_accuracy: 0.8081 - val_loss: 0.4419 - learning_rate: 1.0000e-04
    Epoch 142/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7661 - loss: 0.4876 - val_accuracy: 0.8049 - val_loss: 0.4503 - learning_rate: 1.0000e-04
    Epoch 143/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7761 - loss: 0.4749 - val_accuracy: 0.8103 - val_loss: 0.4504 - learning_rate: 1.0000e-04
    Epoch 144/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7705 - loss: 0.4923 - val_accuracy: 0.7942 - val_loss: 0.4531 - learning_rate: 1.0000e-04
    Epoch 145/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7738 - loss: 0.4890 - val_accuracy: 0.7921 - val_loss: 0.4541 - learning_rate: 1.0000e-04
    Epoch 146/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7696 - loss: 0.4947 - val_accuracy: 0.8028 - val_loss: 0.4487 - learning_rate: 1.0000e-04
    Epoch 147/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7552 - loss: 0.4883 - val_accuracy: 0.7803 - val_loss: 0.4604 - learning_rate: 1.0000e-04
    Epoch 148/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7682 - loss: 0.4809 - val_accuracy: 0.8156 - val_loss: 0.4465 - learning_rate: 1.0000e-04
    Epoch 149/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7665 - loss: 0.4955 - val_accuracy: 0.8081 - val_loss: 0.4468 - learning_rate: 1.0000e-04
    Epoch 150/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7814 - loss: 0.4832 - val_accuracy: 0.7964 - val_loss: 0.4431 - learning_rate: 1.0000e-04
    Epoch 151/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7702 - loss: 0.4834 - val_accuracy: 0.8006 - val_loss: 0.4473 - learning_rate: 1.0000e-04
    Epoch 152/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7740 - loss: 0.4856 - val_accuracy: 0.8006 - val_loss: 0.4520 - learning_rate: 1.0000e-04
    Epoch 153/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7644 - loss: 0.4965 - val_accuracy: 0.7846 - val_loss: 0.4498 - learning_rate: 1.0000e-04
    Epoch 154/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7857 - loss: 0.4678 - val_accuracy: 0.7599 - val_loss: 0.4881 - learning_rate: 1.0000e-04
    Epoch 155/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7743 - loss: 0.4826 - val_accuracy: 0.7942 - val_loss: 0.4487 - learning_rate: 1.0000e-04
    Epoch 156/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7676 - loss: 0.4826 - val_accuracy: 0.7899 - val_loss: 0.4555 - learning_rate: 1.0000e-04
    Epoch 157/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7713 - loss: 0.4790 - val_accuracy: 0.8081 - val_loss: 0.4387 - learning_rate: 1.0000e-04
    Epoch 158/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7774 - loss: 0.4782 - val_accuracy: 0.8124 - val_loss: 0.4413 - learning_rate: 1.0000e-04
    Epoch 159/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7708 - loss: 0.4990 - val_accuracy: 0.8028 - val_loss: 0.4385 - learning_rate: 1.0000e-04
    Epoch 160/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7810 - loss: 0.4860 - val_accuracy: 0.8114 - val_loss: 0.4378 - learning_rate: 1.0000e-04
    Epoch 161/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7745 - loss: 0.4835 - val_accuracy: 0.8124 - val_loss: 0.4430 - learning_rate: 1.0000e-04
    Epoch 162/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7818 - loss: 0.4753 - val_accuracy: 0.7856 - val_loss: 0.4503 - learning_rate: 1.0000e-04
    Epoch 163/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7724 - loss: 0.4810 - val_accuracy: 0.7985 - val_loss: 0.4499 - learning_rate: 1.0000e-04
    Epoch 164/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7654 - loss: 0.4956 - val_accuracy: 0.7953 - val_loss: 0.4535 - learning_rate: 1.0000e-04
    Epoch 165/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7550 - loss: 0.5046 - val_accuracy: 0.8081 - val_loss: 0.4365 - learning_rate: 1.0000e-04
    Epoch 166/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7741 - loss: 0.4819 - val_accuracy: 0.7942 - val_loss: 0.4449 - learning_rate: 1.0000e-04
    Epoch 167/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7684 - loss: 0.4816 - val_accuracy: 0.8006 - val_loss: 0.4454 - learning_rate: 1.0000e-04
    Epoch 168/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7774 - loss: 0.4784 - val_accuracy: 0.8146 - val_loss: 0.4245 - learning_rate: 1.0000e-04
    Epoch 169/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7840 - loss: 0.4832 - val_accuracy: 0.8114 - val_loss: 0.4391 - learning_rate: 1.0000e-04
    Epoch 170/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7658 - loss: 0.4833 - val_accuracy: 0.8103 - val_loss: 0.4282 - learning_rate: 1.0000e-04
    Epoch 171/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7759 - loss: 0.4665 - val_accuracy: 0.7631 - val_loss: 0.4703 - learning_rate: 1.0000e-04
    Epoch 172/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7722 - loss: 0.4823 - val_accuracy: 0.8049 - val_loss: 0.4369 - learning_rate: 1.0000e-04
    Epoch 173/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7795 - loss: 0.4775 - val_accuracy: 0.8114 - val_loss: 0.4217 - learning_rate: 1.0000e-04
    Epoch 174/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7839 - loss: 0.4551 - val_accuracy: 0.8167 - val_loss: 0.4239 - learning_rate: 1.0000e-04
    Epoch 175/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7782 - loss: 0.4653 - val_accuracy: 0.8274 - val_loss: 0.4318 - learning_rate: 1.0000e-04
    Epoch 176/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7856 - loss: 0.4577 - val_accuracy: 0.8135 - val_loss: 0.4335 - learning_rate: 1.0000e-04
    Epoch 177/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7794 - loss: 0.4797 - val_accuracy: 0.8189 - val_loss: 0.4188 - learning_rate: 1.0000e-04
    Epoch 178/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7800 - loss: 0.4512 - val_accuracy: 0.8156 - val_loss: 0.4285 - learning_rate: 1.0000e-04
    Epoch 179/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7743 - loss: 0.4778 - val_accuracy: 0.8156 - val_loss: 0.4330 - learning_rate: 1.0000e-04
    Epoch 180/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7794 - loss: 0.4738 - val_accuracy: 0.8199 - val_loss: 0.4272 - learning_rate: 1.0000e-04
    Epoch 181/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7875 - loss: 0.4631 - val_accuracy: 0.8199 - val_loss: 0.4263 - learning_rate: 1.0000e-04
    Epoch 182/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7817 - loss: 0.4611 - val_accuracy: 0.8039 - val_loss: 0.4243 - learning_rate: 1.0000e-04
    Epoch 183/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7861 - loss: 0.4508 - val_accuracy: 0.8135 - val_loss: 0.4153 - learning_rate: 1.0000e-04
    Epoch 184/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7828 - loss: 0.4670 - val_accuracy: 0.7964 - val_loss: 0.4351 - learning_rate: 1.0000e-04
    Epoch 185/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7883 - loss: 0.4573 - val_accuracy: 0.8242 - val_loss: 0.4157 - learning_rate: 1.0000e-04
    Epoch 186/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7772 - loss: 0.4777 - val_accuracy: 0.8060 - val_loss: 0.4279 - learning_rate: 1.0000e-04
    Epoch 187/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7961 - loss: 0.4574 - val_accuracy: 0.8081 - val_loss: 0.4321 - learning_rate: 1.0000e-04
    Epoch 188/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7880 - loss: 0.4511 - val_accuracy: 0.8264 - val_loss: 0.4218 - learning_rate: 1.0000e-04
    Epoch 189/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7847 - loss: 0.4639 - val_accuracy: 0.7953 - val_loss: 0.4353 - learning_rate: 1.0000e-04
    Epoch 190/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7891 - loss: 0.4468 - val_accuracy: 0.8135 - val_loss: 0.4242 - learning_rate: 1.0000e-04
    Epoch 191/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7921 - loss: 0.4514 - val_accuracy: 0.8156 - val_loss: 0.4064 - learning_rate: 1.0000e-04
    Epoch 192/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.8040 - loss: 0.4481 - val_accuracy: 0.8178 - val_loss: 0.4227 - learning_rate: 1.0000e-04
    Epoch 193/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7813 - loss: 0.4816 - val_accuracy: 0.8285 - val_loss: 0.4154 - learning_rate: 1.0000e-04
    Epoch 194/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7892 - loss: 0.4651 - val_accuracy: 0.8167 - val_loss: 0.4100 - learning_rate: 1.0000e-04
    Epoch 195/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7811 - loss: 0.4677 - val_accuracy: 0.8071 - val_loss: 0.4214 - learning_rate: 1.0000e-04
    Epoch 196/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7883 - loss: 0.4530 - val_accuracy: 0.8167 - val_loss: 0.4233 - learning_rate: 1.0000e-04
    Epoch 197/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7895 - loss: 0.4555 - val_accuracy: 0.8232 - val_loss: 0.4130 - learning_rate: 1.0000e-04
    Epoch 198/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7910 - loss: 0.4449 - val_accuracy: 0.8253 - val_loss: 0.4177 - learning_rate: 1.0000e-04
    Epoch 199/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7835 - loss: 0.4625 - val_accuracy: 0.7931 - val_loss: 0.4277 - learning_rate: 1.0000e-04
    Epoch 200/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.8050 - loss: 0.4465 - val_accuracy: 0.8264 - val_loss: 0.4070 - learning_rate: 1.0000e-04
    Epoch 201/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7956 - loss: 0.4425 - val_accuracy: 0.7856 - val_loss: 0.4433 - learning_rate: 1.0000e-04
    Epoch 202/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7791 - loss: 0.4516 - val_accuracy: 0.8307 - val_loss: 0.4095 - learning_rate: 1.0000e-04
    Epoch 203/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.8006 - loss: 0.4374 - val_accuracy: 0.8264 - val_loss: 0.4145 - learning_rate: 1.0000e-04
    Epoch 204/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.8048 - loss: 0.4436 - val_accuracy: 0.8178 - val_loss: 0.4074 - learning_rate: 1.0000e-04
    Epoch 205/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7994 - loss: 0.4363 - val_accuracy: 0.7814 - val_loss: 0.4597 - learning_rate: 1.0000e-04
    Epoch 206/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7955 - loss: 0.4491 - val_accuracy: 0.8178 - val_loss: 0.4078 - learning_rate: 1.0000e-04
    Epoch 207/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7895 - loss: 0.4371 - val_accuracy: 0.8242 - val_loss: 0.4150 - learning_rate: 1.0000e-04
    Epoch 208/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7959 - loss: 0.4583 - val_accuracy: 0.8307 - val_loss: 0.4008 - learning_rate: 1.0000e-04
    Epoch 209/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.8027 - loss: 0.4349 - val_accuracy: 0.8317 - val_loss: 0.4048 - learning_rate: 1.0000e-04
    Epoch 210/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7983 - loss: 0.4416 - val_accuracy: 0.8103 - val_loss: 0.4164 - learning_rate: 1.0000e-04
    Epoch 211/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.8099 - loss: 0.4303 - val_accuracy: 0.8371 - val_loss: 0.4056 - learning_rate: 1.0000e-04
    Epoch 212/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.8087 - loss: 0.4316 - val_accuracy: 0.8092 - val_loss: 0.4151 - learning_rate: 1.0000e-04
    Epoch 213/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7921 - loss: 0.4533 - val_accuracy: 0.8317 - val_loss: 0.4019 - learning_rate: 1.0000e-04
    Epoch 214/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.8043 - loss: 0.4350 - val_accuracy: 0.8167 - val_loss: 0.4077 - learning_rate: 1.0000e-04
    Epoch 215/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7855 - loss: 0.4604 - val_accuracy: 0.8285 - val_loss: 0.4015 - learning_rate: 1.0000e-04
    Epoch 216/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.8062 - loss: 0.4304 - val_accuracy: 0.8285 - val_loss: 0.4011 - learning_rate: 1.0000e-04
    Epoch 217/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7957 - loss: 0.4414 - val_accuracy: 0.8360 - val_loss: 0.4039 - learning_rate: 1.0000e-04
    Epoch 218/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7975 - loss: 0.4349 - val_accuracy: 0.8328 - val_loss: 0.4013 - learning_rate: 1.0000e-04
    Epoch 219/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.8064 - loss: 0.4379 - val_accuracy: 0.8028 - val_loss: 0.4181 - learning_rate: 1.0000e-04
    Epoch 220/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7870 - loss: 0.4493 - val_accuracy: 0.8307 - val_loss: 0.4015 - learning_rate: 1.0000e-04
    Epoch 221/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7920 - loss: 0.4479 - val_accuracy: 0.8360 - val_loss: 0.3958 - learning_rate: 1.0000e-04
    Epoch 222/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7856 - loss: 0.4563 - val_accuracy: 0.8242 - val_loss: 0.4042 - learning_rate: 1.0000e-04
    Epoch 223/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.8082 - loss: 0.4232 - val_accuracy: 0.8253 - val_loss: 0.3953 - learning_rate: 1.0000e-04
    Epoch 224/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.8033 - loss: 0.4452 - val_accuracy: 0.8264 - val_loss: 0.3991 - learning_rate: 1.0000e-04
    Epoch 225/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.8128 - loss: 0.4171 - val_accuracy: 0.8242 - val_loss: 0.4054 - learning_rate: 1.0000e-04
    Epoch 226/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7957 - loss: 0.4422 - val_accuracy: 0.8221 - val_loss: 0.4003 - learning_rate: 1.0000e-04
    Epoch 227/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.8090 - loss: 0.4264 - val_accuracy: 0.8339 - val_loss: 0.3975 - learning_rate: 1.0000e-04
    Epoch 228/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 12ms/step - accuracy: 0.8071 - loss: 0.4329 - val_accuracy: 0.8221 - val_loss: 0.3977 - learning_rate: 1.0000e-04
    Epoch 229/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.8003 - loss: 0.4319 - val_accuracy: 0.8274 - val_loss: 0.3898 - learning_rate: 1.0000e-04
    Epoch 230/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.8015 - loss: 0.4346 - val_accuracy: 0.8146 - val_loss: 0.4085 - learning_rate: 1.0000e-04
    Epoch 231/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7905 - loss: 0.4344 - val_accuracy: 0.8371 - val_loss: 0.3951 - learning_rate: 1.0000e-04
    Epoch 232/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7985 - loss: 0.4396 - val_accuracy: 0.8328 - val_loss: 0.4040 - learning_rate: 1.0000e-04
    Epoch 233/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.8099 - loss: 0.4217 - val_accuracy: 0.8382 - val_loss: 0.3894 - learning_rate: 1.0000e-04
    Epoch 234/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7971 - loss: 0.4401 - val_accuracy: 0.8392 - val_loss: 0.3939 - learning_rate: 1.0000e-04
    Epoch 235/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.8054 - loss: 0.4271 - val_accuracy: 0.8071 - val_loss: 0.4178 - learning_rate: 1.0000e-04
    Epoch 236/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.8055 - loss: 0.4347 - val_accuracy: 0.8264 - val_loss: 0.4077 - learning_rate: 1.0000e-04
    Epoch 237/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.8120 - loss: 0.4236 - val_accuracy: 0.8349 - val_loss: 0.3888 - learning_rate: 1.0000e-04
    Epoch 238/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.8086 - loss: 0.4239 - val_accuracy: 0.8349 - val_loss: 0.3958 - learning_rate: 1.0000e-04
    Epoch 239/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7989 - loss: 0.4348 - val_accuracy: 0.8392 - val_loss: 0.3925 - learning_rate: 1.0000e-04
    Epoch 240/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.8070 - loss: 0.4302 - val_accuracy: 0.8264 - val_loss: 0.3987 - learning_rate: 1.0000e-04
    Epoch 241/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.8044 - loss: 0.4312 - val_accuracy: 0.8435 - val_loss: 0.3870 - learning_rate: 1.0000e-04
    Epoch 242/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.8169 - loss: 0.4115 - val_accuracy: 0.8339 - val_loss: 0.3970 - learning_rate: 1.0000e-04
    Epoch 243/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.8017 - loss: 0.4287 - val_accuracy: 0.8274 - val_loss: 0.3913 - learning_rate: 1.0000e-04
    Epoch 244/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.7910 - loss: 0.4426 - val_accuracy: 0.8446 - val_loss: 0.3875 - learning_rate: 1.0000e-04
    Epoch 245/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.8055 - loss: 0.4329 - val_accuracy: 0.7996 - val_loss: 0.4276 - learning_rate: 1.0000e-04
    Epoch 246/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.8146 - loss: 0.4261 - val_accuracy: 0.8403 - val_loss: 0.3880 - learning_rate: 1.0000e-04
    Epoch 247/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 12ms/step - accuracy: 0.7991 - loss: 0.4301 - val_accuracy: 0.8274 - val_loss: 0.3910 - learning_rate: 1.0000e-04
    Epoch 248/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.8087 - loss: 0.4301 - val_accuracy: 0.8360 - val_loss: 0.3883 - learning_rate: 1.0000e-04
    Epoch 249/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 11ms/step - accuracy: 0.8126 - loss: 0.4226 - val_accuracy: 0.8210 - val_loss: 0.4035 - learning_rate: 1.0000e-04
    Epoch 250/250
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 12ms/step - accuracy: 0.8124 - loss: 0.4296 - val_accuracy: 0.8403 - val_loss: 0.3922 - learning_rate: 1.0000e-04
    Restoring model weights from the end of the best epoch: 241.



```python
# Create dataframe of model fit history.
mod_fnn_hist_df = pd.DataFrame().from_dict(mod_fnn_hist.history, orient = 'columns')

plot_TF_training_history(mod_fnn_hist_df, 'KNN - Training History')
```


    
![png](Periocular_Sex_Classification_files/Periocular_Sex_Classification_67_0.png)
    


This looks great, and the model didn't overfit to the training data. The model seems slightly unstable, but that is to be expected especially with the random augmentations being applied to the training data at each epoch because this type of model is not invariant to the types of augmentations being performed.

Looking at the training history here, the model could likely be trainied for a few more epochs, but through tuning it I know that this is about as stable as it gets and has essentially converged here so we will move on.


```python
y_pred_val_fnn_proba = mod_fnn.predict(ds_val, verbose = "auto", callbacks = None)
y_pred_val_fnn = y_pred_val_fnn_proba.argmax(axis = 1)

accuracy_score(y_val, y_pred_val_fnn)
```

    [1m59/59[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step





    0.8435155412647374



84.35% accuracy on the validation set is a great score for image classification using only dense layers!

Now the prediction on the test set will be performed and saved for the results section after all tuning is completed.


```python
y_pred_fnn_proba = mod_fnn.predict(np.stack(X_test.norm_flat.apply(flat_to_array)), verbose = "auto", callbacks = None)
y_pred_fnn = y_pred_fnn_proba.argmax(axis = 1)
```

    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step


##### Non-Augmented Training

Train the same model but use the training set that is not augmented/transformed for comparison at the end. Muted outputs for notebook clarity.


```python
mod_fnn_no_aug = build_fnn()
mod_fnn_no_aug.compile(loss = loss_fx,
                optimizer = optimizer_param,
                metrics = ['accuracy'])

# Callbacks to use (same as above):
early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 25, verbose = 1, restore_best_weights = True)
reduce_lr_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 5, cooldown = 8, min_lr = 0.0001, verbose = 0)

mod_fnn_no_aug_hist = mod_fnn_no_aug.fit(ds_train,
                           batch_size = BATCH_SIZE,
                           epochs = n_epochs,
                           verbose = 0,
                           callbacks = [early_stop, reduce_lr_plateau],
                           validation_split = 0.0,
                           validation_data = ds_val,
                           shuffle = True,
                           class_weight = None,
                           sample_weight = None,
                           initial_epoch = 0,
                           steps_per_epoch = None,
                           validation_steps = None,
                           validation_batch_size = None,
                           validation_freq = val_freq)

y_pred_val_fnn_no_aug_proba = mod_fnn_no_aug.predict(ds_val, verbose = 0, callbacks = None)
y_pred_val_fnn_no_aug = y_pred_val_fnn_no_aug_proba.argmax(axis = 1)

y_pred_fnn_no_aug_proba = mod_fnn_no_aug.predict(np.stack(X_test.norm_flat.apply(flat_to_array)), verbose = 0, callbacks = None)
y_pred_fnn_no_aug = y_pred_fnn_no_aug_proba.argmax(axis = 1)
```

    Epoch 100: early stopping
    Restoring model weights from the end of the best epoch: 75.


We can see training without the variance that randomly augmenting introduces into the dataset, heavily reduces the training time. This model reverted back to epoch 75 while the previous model, with image augmentations, trained until 241. We'll see how this effects performance in the results section.

###### [Back to Table of Contents](#toc)

#### 7.2.2. Deep-ish Convolutional Neural Network (CNN): <a name="cnn"></a>

Model Specifications:

- Loss Function: Sparse Categorical using Cross-Entropy
- Optimizer: Stochastic Gradient Descent (SGD)
- Callbacks:
    - Early Stopping
    - Adjust Learning Rate on Loss Plateau

The model below showcases the benefits of stacking multiple 3x3 convolutional layers instead of utilizing larger kernel sizes. 

Consider a network using the same number of filters with larger kernel sizes (such as 32 filters of 5x5 and 96 filters of 7x7) would require 20,473,314 trainable parameters. This model, by distributing the same number of filters across multiple layers of 3x3 kernels, achieves the same receptive field with only 7,501,106 parameters. This significant reduction is due to the fact that larger kernels introduces redundant parameter calculations without adding unique value to the model's learning capability.

An additional advantage of using multiple 3x3 kernels is the introduction of activation functions between layers, which increases the model’s non-linearity and allows it to capture more complex patterns in the data compared to larger single-layer kernels.

These factors allow the use of much deeper networks while maintaining computational efficiency, which is why modern architectures predominantly use stacked 3x3 kernels.


```python
def build_cnn(input_shape = (img_size, img_size, 3)):

    model = tf.keras.models.Sequential([
        tf.keras.Input(shape = input_shape, name = 'Image Array'),
        # First block (2).
        tf.keras.layers.Conv2D(16, kernel_size = (3,3), padding = 'same'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(16, kernel_size = (3,3), padding = 'same'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2,2), strides = 2),

        # Second block (3)
        tf.keras.layers.Conv2D(32, kernel_size = (3,3), padding = 'same'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(32, kernel_size = (3,3), padding = 'same'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(32, kernel_size = (3,3), padding = 'same'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2,2), strides = 2),

        # Classifier block (2).
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1024),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, name = 'Predictions')])
    return model

mod_cnn = build_cnn()
mod_cnn.summary()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential_3"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)     │           <span style="color: #00af00; text-decoration-color: #00af00">448</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ re_lu_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)                  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)     │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)     │         <span style="color: #00af00; text-decoration-color: #00af00">2,320</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ re_lu_5 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)                  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)     │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)     │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)     │         <span style="color: #00af00; text-decoration-color: #00af00">4,640</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ re_lu_6 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)                  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)     │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)     │         <span style="color: #00af00; text-decoration-color: #00af00">9,248</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ re_lu_7 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)                  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)     │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)     │         <span style="color: #00af00; text-decoration-color: #00af00">9,248</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ re_lu_8 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)                  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)     │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)     │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">6272</span>)           │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)           │     <span style="color: #00af00; text-decoration-color: #00af00">6,423,552</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ re_lu_9 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)                  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)           │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)           │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_5 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)           │     <span style="color: #00af00; text-decoration-color: #00af00">1,049,600</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ re_lu_10 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)           │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_5 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)           │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Predictions (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>)              │         <span style="color: #00af00; text-decoration-color: #00af00">2,050</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">7,501,106</span> (28.61 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">7,501,106</span> (28.61 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>




```python
visualkeras.layered_view(mod_cnn, legend = True)
```




    
![png](Periocular_Sex_Classification_files/Periocular_Sex_Classification_79_0.png)
    




```python
loss_fx = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True) # Integer labels.
optimizer_param = 'sgd'
val_freq = 1
n_epochs = 300
# Callbacks to use:
early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 40, verbose = 1, restore_best_weights = True)
reduce_lr_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', 
                                                         factor = 0.5,
                                                         patience = 5,
                                                         cooldown = 8,
                                                         min_lr = 0.0001,
                                                         verbose = 1)
```


```python
mod_cnn.compile(loss = loss_fx,
                optimizer = optimizer_param,
                metrics = ['accuracy'])

mod_cnn_hist = mod_cnn.fit(ds_train_transformed,
                           batch_size = BATCH_SIZE,
                           epochs = n_epochs,
                           verbose = "auto",
                           callbacks = [early_stop, reduce_lr_plateau],
                           validation_split = 0.0,
                           validation_data = ds_val,
                           shuffle = True,
                           class_weight = None,
                           sample_weight = None,
                           initial_epoch = 0,
                           steps_per_epoch = None,
                           validation_steps = None,
                           validation_batch_size = None,
                           validation_freq = val_freq)
```

    Epoch 1/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 31ms/step - accuracy: 0.5011 - loss: 0.6938 - val_accuracy: 0.5102 - val_loss: 0.6913 - learning_rate: 0.0100
    Epoch 2/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.5154 - loss: 0.6925 - val_accuracy: 0.5852 - val_loss: 0.6893 - learning_rate: 0.0100
    Epoch 3/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.5191 - loss: 0.6922 - val_accuracy: 0.6120 - val_loss: 0.6847 - learning_rate: 0.0100
    Epoch 4/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.5469 - loss: 0.6889 - val_accuracy: 0.6452 - val_loss: 0.6814 - learning_rate: 0.0100
    Epoch 5/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.5495 - loss: 0.6881 - val_accuracy: 0.6710 - val_loss: 0.6748 - learning_rate: 0.0100
    Epoch 6/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.5820 - loss: 0.6793 - val_accuracy: 0.7063 - val_loss: 0.6572 - learning_rate: 0.0100
    Epoch 7/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.6005 - loss: 0.6722 - val_accuracy: 0.7631 - val_loss: 0.6198 - learning_rate: 0.0100
    Epoch 8/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.6253 - loss: 0.6511 - val_accuracy: 0.7149 - val_loss: 0.5777 - learning_rate: 0.0100
    Epoch 9/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.6540 - loss: 0.6289 - val_accuracy: 0.7621 - val_loss: 0.5655 - learning_rate: 0.0100
    Epoch 10/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.6820 - loss: 0.5970 - val_accuracy: 0.7567 - val_loss: 0.5013 - learning_rate: 0.0100
    Epoch 11/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.6841 - loss: 0.5850 - val_accuracy: 0.7760 - val_loss: 0.4885 - learning_rate: 0.0100
    Epoch 12/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.6954 - loss: 0.5744 - val_accuracy: 0.7792 - val_loss: 0.4834 - learning_rate: 0.0100
    Epoch 13/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.7263 - loss: 0.5557 - val_accuracy: 0.7985 - val_loss: 0.4540 - learning_rate: 0.0100
    Epoch 14/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.7243 - loss: 0.5405 - val_accuracy: 0.7942 - val_loss: 0.4566 - learning_rate: 0.0100
    Epoch 15/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.7226 - loss: 0.5424 - val_accuracy: 0.7320 - val_loss: 0.5362 - learning_rate: 0.0100
    Epoch 16/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.7399 - loss: 0.5198 - val_accuracy: 0.7567 - val_loss: 0.4724 - learning_rate: 0.0100
    Epoch 17/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 31ms/step - accuracy: 0.7350 - loss: 0.5234 - val_accuracy: 0.8135 - val_loss: 0.4254 - learning_rate: 0.0100
    Epoch 18/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.7470 - loss: 0.5144 - val_accuracy: 0.7921 - val_loss: 0.4480 - learning_rate: 0.0100
    Epoch 19/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.7591 - loss: 0.5013 - val_accuracy: 0.8253 - val_loss: 0.4139 - learning_rate: 0.0100
    Epoch 20/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.7626 - loss: 0.4899 - val_accuracy: 0.8274 - val_loss: 0.4054 - learning_rate: 0.0100
    Epoch 21/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.7649 - loss: 0.4918 - val_accuracy: 0.8071 - val_loss: 0.4250 - learning_rate: 0.0100
    Epoch 22/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 31ms/step - accuracy: 0.7655 - loss: 0.4863 - val_accuracy: 0.8167 - val_loss: 0.4035 - learning_rate: 0.0100
    Epoch 23/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.7694 - loss: 0.4739 - val_accuracy: 0.8478 - val_loss: 0.3720 - learning_rate: 0.0100
    Epoch 24/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.7823 - loss: 0.4741 - val_accuracy: 0.8414 - val_loss: 0.3547 - learning_rate: 0.0100
    Epoch 25/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.7871 - loss: 0.4499 - val_accuracy: 0.8639 - val_loss: 0.3458 - learning_rate: 0.0100
    Epoch 26/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.7864 - loss: 0.4616 - val_accuracy: 0.8542 - val_loss: 0.3329 - learning_rate: 0.0100
    Epoch 27/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.7856 - loss: 0.4494 - val_accuracy: 0.8628 - val_loss: 0.3377 - learning_rate: 0.0100
    Epoch 28/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.7922 - loss: 0.4409 - val_accuracy: 0.8542 - val_loss: 0.3508 - learning_rate: 0.0100
    Epoch 29/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.8033 - loss: 0.4319 - val_accuracy: 0.8457 - val_loss: 0.3438 - learning_rate: 0.0100
    Epoch 30/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.7900 - loss: 0.4338 - val_accuracy: 0.8789 - val_loss: 0.3046 - learning_rate: 0.0100
    Epoch 31/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.8133 - loss: 0.4117 - val_accuracy: 0.8725 - val_loss: 0.2968 - learning_rate: 0.0100
    Epoch 32/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.7944 - loss: 0.4390 - val_accuracy: 0.8757 - val_loss: 0.2964 - learning_rate: 0.0100
    Epoch 33/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.8291 - loss: 0.4056 - val_accuracy: 0.8424 - val_loss: 0.3535 - learning_rate: 0.0100
    Epoch 34/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.8314 - loss: 0.3855 - val_accuracy: 0.8778 - val_loss: 0.2943 - learning_rate: 0.0100
    Epoch 35/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 31ms/step - accuracy: 0.8138 - loss: 0.4047 - val_accuracy: 0.8950 - val_loss: 0.2706 - learning_rate: 0.0100
    Epoch 36/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.8231 - loss: 0.3887 - val_accuracy: 0.8917 - val_loss: 0.2744 - learning_rate: 0.0100
    Epoch 37/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.8256 - loss: 0.3958 - val_accuracy: 0.8778 - val_loss: 0.2995 - learning_rate: 0.0100
    Epoch 38/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.8426 - loss: 0.3588 - val_accuracy: 0.8917 - val_loss: 0.2440 - learning_rate: 0.0100
    Epoch 39/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.8346 - loss: 0.3701 - val_accuracy: 0.8907 - val_loss: 0.2743 - learning_rate: 0.0100
    Epoch 40/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.8373 - loss: 0.3659 - val_accuracy: 0.9164 - val_loss: 0.2297 - learning_rate: 0.0100
    Epoch 41/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.8382 - loss: 0.3633 - val_accuracy: 0.9025 - val_loss: 0.2460 - learning_rate: 0.0100
    Epoch 42/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.8512 - loss: 0.3483 - val_accuracy: 0.9100 - val_loss: 0.2303 - learning_rate: 0.0100
    Epoch 43/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.8413 - loss: 0.3498 - val_accuracy: 0.8585 - val_loss: 0.3174 - learning_rate: 0.0100
    Epoch 44/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.8450 - loss: 0.3560 - val_accuracy: 0.9046 - val_loss: 0.2404 - learning_rate: 0.0100
    Epoch 45/300
    [1m226/227[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 30ms/step - accuracy: 0.8476 - loss: 0.3401
    Epoch 45: ReduceLROnPlateau reducing learning rate to 0.004999999888241291.
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.8476 - loss: 0.3401 - val_accuracy: 0.9014 - val_loss: 0.2374 - learning_rate: 0.0100
    Epoch 46/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.8725 - loss: 0.3162 - val_accuracy: 0.9132 - val_loss: 0.2332 - learning_rate: 0.0050
    Epoch 47/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.8632 - loss: 0.3204 - val_accuracy: 0.9218 - val_loss: 0.2122 - learning_rate: 0.0050
    Epoch 48/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.8713 - loss: 0.2996 - val_accuracy: 0.9078 - val_loss: 0.2127 - learning_rate: 0.0050
    Epoch 49/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.8519 - loss: 0.3181 - val_accuracy: 0.9121 - val_loss: 0.2198 - learning_rate: 0.0050
    Epoch 50/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.8679 - loss: 0.3034 - val_accuracy: 0.9143 - val_loss: 0.2086 - learning_rate: 0.0050
    Epoch 51/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.8655 - loss: 0.3207 - val_accuracy: 0.9175 - val_loss: 0.2065 - learning_rate: 0.0050
    Epoch 52/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 31ms/step - accuracy: 0.8789 - loss: 0.2945 - val_accuracy: 0.8746 - val_loss: 0.2998 - learning_rate: 0.0050
    Epoch 53/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 31ms/step - accuracy: 0.8767 - loss: 0.2996 - val_accuracy: 0.9196 - val_loss: 0.1997 - learning_rate: 0.0050
    Epoch 54/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.8786 - loss: 0.2854 - val_accuracy: 0.9250 - val_loss: 0.1888 - learning_rate: 0.0050
    Epoch 55/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 31ms/step - accuracy: 0.8765 - loss: 0.2987 - val_accuracy: 0.9196 - val_loss: 0.1927 - learning_rate: 0.0050
    Epoch 56/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.8754 - loss: 0.2901 - val_accuracy: 0.9132 - val_loss: 0.2104 - learning_rate: 0.0050
    Epoch 57/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.8884 - loss: 0.2794 - val_accuracy: 0.9185 - val_loss: 0.1992 - learning_rate: 0.0050
    Epoch 58/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.8749 - loss: 0.2927 - val_accuracy: 0.9175 - val_loss: 0.1999 - learning_rate: 0.0050
    Epoch 59/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.8772 - loss: 0.2900 - val_accuracy: 0.9239 - val_loss: 0.1839 - learning_rate: 0.0050
    Epoch 60/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.8814 - loss: 0.2912 - val_accuracy: 0.9089 - val_loss: 0.2196 - learning_rate: 0.0050
    Epoch 61/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.8890 - loss: 0.2832 - val_accuracy: 0.9185 - val_loss: 0.1918 - learning_rate: 0.0050
    Epoch 62/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.8780 - loss: 0.2926 - val_accuracy: 0.9282 - val_loss: 0.1931 - learning_rate: 0.0050
    Epoch 63/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.8823 - loss: 0.2797 - val_accuracy: 0.9143 - val_loss: 0.2004 - learning_rate: 0.0050
    Epoch 64/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.8863 - loss: 0.2766 - val_accuracy: 0.9196 - val_loss: 0.1817 - learning_rate: 0.0050
    Epoch 65/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.8914 - loss: 0.2683 - val_accuracy: 0.9185 - val_loss: 0.1980 - learning_rate: 0.0050
    Epoch 66/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.8794 - loss: 0.2799 - val_accuracy: 0.9153 - val_loss: 0.1942 - learning_rate: 0.0050
    Epoch 67/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.8887 - loss: 0.2684 - val_accuracy: 0.9250 - val_loss: 0.1917 - learning_rate: 0.0050
    Epoch 68/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.8917 - loss: 0.2583 - val_accuracy: 0.9239 - val_loss: 0.1793 - learning_rate: 0.0050
    Epoch 69/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.9003 - loss: 0.2638 - val_accuracy: 0.9282 - val_loss: 0.1779 - learning_rate: 0.0050
    Epoch 70/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.8830 - loss: 0.2740 - val_accuracy: 0.9293 - val_loss: 0.1807 - learning_rate: 0.0050
    Epoch 71/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.8905 - loss: 0.2580 - val_accuracy: 0.9357 - val_loss: 0.1728 - learning_rate: 0.0050
    Epoch 72/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.8900 - loss: 0.2610 - val_accuracy: 0.9078 - val_loss: 0.2174 - learning_rate: 0.0050
    Epoch 73/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.8875 - loss: 0.2769 - val_accuracy: 0.9335 - val_loss: 0.1798 - learning_rate: 0.0050
    Epoch 74/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.8993 - loss: 0.2427 - val_accuracy: 0.9293 - val_loss: 0.1737 - learning_rate: 0.0050
    Epoch 75/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.8943 - loss: 0.2575 - val_accuracy: 0.9293 - val_loss: 0.1723 - learning_rate: 0.0050
    Epoch 76/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.8883 - loss: 0.2710 - val_accuracy: 0.9314 - val_loss: 0.1756 - learning_rate: 0.0050
    Epoch 77/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.8879 - loss: 0.2689 - val_accuracy: 0.9143 - val_loss: 0.1914 - learning_rate: 0.0050
    Epoch 78/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.8976 - loss: 0.2505 - val_accuracy: 0.9271 - val_loss: 0.1732 - learning_rate: 0.0050
    Epoch 79/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.8868 - loss: 0.2580 - val_accuracy: 0.9207 - val_loss: 0.1936 - learning_rate: 0.0050
    Epoch 80/300
    [1m226/227[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 29ms/step - accuracy: 0.9006 - loss: 0.2337
    Epoch 80: ReduceLROnPlateau reducing learning rate to 0.0024999999441206455.
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9006 - loss: 0.2338 - val_accuracy: 0.9335 - val_loss: 0.1901 - learning_rate: 0.0050
    Epoch 81/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9058 - loss: 0.2443 - val_accuracy: 0.9400 - val_loss: 0.1680 - learning_rate: 0.0025
    Epoch 82/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9059 - loss: 0.2497 - val_accuracy: 0.9271 - val_loss: 0.1787 - learning_rate: 0.0025
    Epoch 83/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.8978 - loss: 0.2499 - val_accuracy: 0.9357 - val_loss: 0.1680 - learning_rate: 0.0025
    Epoch 84/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9144 - loss: 0.2263 - val_accuracy: 0.9207 - val_loss: 0.1822 - learning_rate: 0.0025
    Epoch 85/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9060 - loss: 0.2320 - val_accuracy: 0.9335 - val_loss: 0.1759 - learning_rate: 0.0025
    Epoch 86/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9049 - loss: 0.2328 - val_accuracy: 0.9378 - val_loss: 0.1650 - learning_rate: 0.0025
    Epoch 87/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9051 - loss: 0.2347 - val_accuracy: 0.9293 - val_loss: 0.1752 - learning_rate: 0.0025
    Epoch 88/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9081 - loss: 0.2220 - val_accuracy: 0.9303 - val_loss: 0.1744 - learning_rate: 0.0025
    Epoch 89/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.8998 - loss: 0.2385 - val_accuracy: 0.9325 - val_loss: 0.1779 - learning_rate: 0.0025
    Epoch 90/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9048 - loss: 0.2296 - val_accuracy: 0.9250 - val_loss: 0.1777 - learning_rate: 0.0025
    Epoch 91/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9115 - loss: 0.2244 - val_accuracy: 0.9303 - val_loss: 0.1718 - learning_rate: 0.0025
    Epoch 92/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - accuracy: 0.8981 - loss: 0.2377
    Epoch 92: ReduceLROnPlateau reducing learning rate to 0.0012499999720603228.
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.8981 - loss: 0.2377 - val_accuracy: 0.9228 - val_loss: 0.1749 - learning_rate: 0.0025
    Epoch 93/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.9005 - loss: 0.2319 - val_accuracy: 0.9293 - val_loss: 0.1833 - learning_rate: 0.0012
    Epoch 94/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9185 - loss: 0.2157 - val_accuracy: 0.9325 - val_loss: 0.1639 - learning_rate: 0.0012
    Epoch 95/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9100 - loss: 0.2236 - val_accuracy: 0.9325 - val_loss: 0.1610 - learning_rate: 0.0012
    Epoch 96/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9136 - loss: 0.2208 - val_accuracy: 0.9314 - val_loss: 0.1637 - learning_rate: 0.0012
    Epoch 97/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9165 - loss: 0.2130 - val_accuracy: 0.9335 - val_loss: 0.1872 - learning_rate: 0.0012
    Epoch 98/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.9192 - loss: 0.2042 - val_accuracy: 0.9368 - val_loss: 0.1552 - learning_rate: 0.0012
    Epoch 99/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9078 - loss: 0.2229 - val_accuracy: 0.9357 - val_loss: 0.1645 - learning_rate: 0.0012
    Epoch 100/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9058 - loss: 0.2288 - val_accuracy: 0.9346 - val_loss: 0.1648 - learning_rate: 0.0012
    Epoch 101/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 31ms/step - accuracy: 0.9139 - loss: 0.2095 - val_accuracy: 0.9303 - val_loss: 0.1745 - learning_rate: 0.0012
    Epoch 102/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9172 - loss: 0.2083 - val_accuracy: 0.9357 - val_loss: 0.1602 - learning_rate: 0.0012
    Epoch 103/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9076 - loss: 0.2213 - val_accuracy: 0.9346 - val_loss: 0.1700 - learning_rate: 0.0012
    Epoch 104/300
    [1m226/227[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 29ms/step - accuracy: 0.9139 - loss: 0.2243
    Epoch 104: ReduceLROnPlateau reducing learning rate to 0.0006249999860301614.
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9139 - loss: 0.2241 - val_accuracy: 0.9432 - val_loss: 0.1580 - learning_rate: 0.0012
    Epoch 105/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.9044 - loss: 0.2167 - val_accuracy: 0.9368 - val_loss: 0.1582 - learning_rate: 6.2500e-04
    Epoch 106/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9060 - loss: 0.2190 - val_accuracy: 0.9346 - val_loss: 0.1588 - learning_rate: 6.2500e-04
    Epoch 107/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9257 - loss: 0.1957 - val_accuracy: 0.9411 - val_loss: 0.1572 - learning_rate: 6.2500e-04
    Epoch 108/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9143 - loss: 0.2089 - val_accuracy: 0.9378 - val_loss: 0.1607 - learning_rate: 6.2500e-04
    Epoch 109/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9148 - loss: 0.2041 - val_accuracy: 0.9400 - val_loss: 0.1572 - learning_rate: 6.2500e-04
    Epoch 110/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9247 - loss: 0.1999 - val_accuracy: 0.9432 - val_loss: 0.1548 - learning_rate: 6.2500e-04
    Epoch 111/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9148 - loss: 0.2049 - val_accuracy: 0.9411 - val_loss: 0.1559 - learning_rate: 6.2500e-04
    Epoch 112/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9098 - loss: 0.2224 - val_accuracy: 0.9400 - val_loss: 0.1604 - learning_rate: 6.2500e-04
    Epoch 113/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.9176 - loss: 0.2087 - val_accuracy: 0.9443 - val_loss: 0.1574 - learning_rate: 6.2500e-04
    Epoch 114/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9219 - loss: 0.2075 - val_accuracy: 0.9411 - val_loss: 0.1580 - learning_rate: 6.2500e-04
    Epoch 115/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9095 - loss: 0.2170 - val_accuracy: 0.9443 - val_loss: 0.1591 - learning_rate: 6.2500e-04
    Epoch 116/300
    [1m226/227[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 30ms/step - accuracy: 0.9250 - loss: 0.1910
    Epoch 116: ReduceLROnPlateau reducing learning rate to 0.0003124999930150807.
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9250 - loss: 0.1910 - val_accuracy: 0.9443 - val_loss: 0.1607 - learning_rate: 6.2500e-04
    Epoch 117/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.9159 - loss: 0.2129 - val_accuracy: 0.9421 - val_loss: 0.1563 - learning_rate: 3.1250e-04
    Epoch 118/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9225 - loss: 0.1998 - val_accuracy: 0.9411 - val_loss: 0.1603 - learning_rate: 3.1250e-04
    Epoch 119/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9168 - loss: 0.2067 - val_accuracy: 0.9368 - val_loss: 0.1586 - learning_rate: 3.1250e-04
    Epoch 120/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9029 - loss: 0.2201 - val_accuracy: 0.9357 - val_loss: 0.1620 - learning_rate: 3.1250e-04
    Epoch 121/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.9237 - loss: 0.1910 - val_accuracy: 0.9400 - val_loss: 0.1601 - learning_rate: 3.1250e-04
    Epoch 122/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9252 - loss: 0.2046 - val_accuracy: 0.9389 - val_loss: 0.1582 - learning_rate: 3.1250e-04
    Epoch 123/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9134 - loss: 0.2142 - val_accuracy: 0.9411 - val_loss: 0.1568 - learning_rate: 3.1250e-04
    Epoch 124/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9092 - loss: 0.2057 - val_accuracy: 0.9389 - val_loss: 0.1554 - learning_rate: 3.1250e-04
    Epoch 125/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9237 - loss: 0.1954 - val_accuracy: 0.9432 - val_loss: 0.1567 - learning_rate: 3.1250e-04
    Epoch 126/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9279 - loss: 0.1928 - val_accuracy: 0.9421 - val_loss: 0.1550 - learning_rate: 3.1250e-04
    Epoch 127/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9282 - loss: 0.1977 - val_accuracy: 0.9411 - val_loss: 0.1560 - learning_rate: 3.1250e-04
    Epoch 128/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step - accuracy: 0.9109 - loss: 0.2052
    Epoch 128: ReduceLROnPlateau reducing learning rate to 0.00015624999650754035.
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9109 - loss: 0.2051 - val_accuracy: 0.9421 - val_loss: 0.1583 - learning_rate: 3.1250e-04
    Epoch 129/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9149 - loss: 0.2078 - val_accuracy: 0.9421 - val_loss: 0.1593 - learning_rate: 1.5625e-04
    Epoch 130/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9188 - loss: 0.1984 - val_accuracy: 0.9368 - val_loss: 0.1595 - learning_rate: 1.5625e-04
    Epoch 131/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9161 - loss: 0.1998 - val_accuracy: 0.9421 - val_loss: 0.1576 - learning_rate: 1.5625e-04
    Epoch 132/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9159 - loss: 0.2001 - val_accuracy: 0.9453 - val_loss: 0.1575 - learning_rate: 1.5625e-04
    Epoch 133/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9134 - loss: 0.2151 - val_accuracy: 0.9421 - val_loss: 0.1587 - learning_rate: 1.5625e-04
    Epoch 134/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9028 - loss: 0.2401 - val_accuracy: 0.9432 - val_loss: 0.1575 - learning_rate: 1.5625e-04
    Epoch 135/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9223 - loss: 0.1980 - val_accuracy: 0.9443 - val_loss: 0.1578 - learning_rate: 1.5625e-04
    Epoch 136/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9255 - loss: 0.1923 - val_accuracy: 0.9443 - val_loss: 0.1565 - learning_rate: 1.5625e-04
    Epoch 137/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9278 - loss: 0.1983 - val_accuracy: 0.9432 - val_loss: 0.1573 - learning_rate: 1.5625e-04
    Epoch 138/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9265 - loss: 0.1852 - val_accuracy: 0.9400 - val_loss: 0.1593 - learning_rate: 1.5625e-04
    Epoch 139/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9230 - loss: 0.1943 - val_accuracy: 0.9453 - val_loss: 0.1550 - learning_rate: 1.5625e-04
    Epoch 140/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - accuracy: 0.9138 - loss: 0.2101
    Epoch 140: ReduceLROnPlateau reducing learning rate to 0.0001.
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 31ms/step - accuracy: 0.9139 - loss: 0.2100 - val_accuracy: 0.9443 - val_loss: 0.1551 - learning_rate: 1.5625e-04
    Epoch 141/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 31ms/step - accuracy: 0.9158 - loss: 0.1997 - val_accuracy: 0.9453 - val_loss: 0.1559 - learning_rate: 1.0000e-04
    Epoch 142/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9273 - loss: 0.1824 - val_accuracy: 0.9432 - val_loss: 0.1573 - learning_rate: 1.0000e-04
    Epoch 143/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.9241 - loss: 0.1929 - val_accuracy: 0.9432 - val_loss: 0.1561 - learning_rate: 1.0000e-04
    Epoch 144/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9091 - loss: 0.2150 - val_accuracy: 0.9421 - val_loss: 0.1557 - learning_rate: 1.0000e-04
    Epoch 145/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9184 - loss: 0.1891 - val_accuracy: 0.9453 - val_loss: 0.1558 - learning_rate: 1.0000e-04
    Epoch 146/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9138 - loss: 0.2102 - val_accuracy: 0.9443 - val_loss: 0.1562 - learning_rate: 1.0000e-04
    Epoch 147/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9170 - loss: 0.1964 - val_accuracy: 0.9443 - val_loss: 0.1559 - learning_rate: 1.0000e-04
    Epoch 148/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9155 - loss: 0.2053 - val_accuracy: 0.9443 - val_loss: 0.1548 - learning_rate: 1.0000e-04
    Epoch 149/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.9191 - loss: 0.1953 - val_accuracy: 0.9443 - val_loss: 0.1565 - learning_rate: 1.0000e-04
    Epoch 150/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9145 - loss: 0.2035 - val_accuracy: 0.9421 - val_loss: 0.1567 - learning_rate: 1.0000e-04
    Epoch 151/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9251 - loss: 0.1833 - val_accuracy: 0.9411 - val_loss: 0.1566 - learning_rate: 1.0000e-04
    Epoch 152/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9224 - loss: 0.1890 - val_accuracy: 0.9443 - val_loss: 0.1565 - learning_rate: 1.0000e-04
    Epoch 153/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9220 - loss: 0.1931 - val_accuracy: 0.9432 - val_loss: 0.1565 - learning_rate: 1.0000e-04
    Epoch 154/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9175 - loss: 0.2009 - val_accuracy: 0.9432 - val_loss: 0.1568 - learning_rate: 1.0000e-04
    Epoch 155/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9164 - loss: 0.1907 - val_accuracy: 0.9464 - val_loss: 0.1552 - learning_rate: 1.0000e-04
    Epoch 156/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.9248 - loss: 0.1993 - val_accuracy: 0.9432 - val_loss: 0.1568 - learning_rate: 1.0000e-04
    Epoch 157/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9198 - loss: 0.1976 - val_accuracy: 0.9432 - val_loss: 0.1573 - learning_rate: 1.0000e-04
    Epoch 158/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9185 - loss: 0.1963 - val_accuracy: 0.9411 - val_loss: 0.1566 - learning_rate: 1.0000e-04
    Epoch 159/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9220 - loss: 0.1838 - val_accuracy: 0.9464 - val_loss: 0.1559 - learning_rate: 1.0000e-04
    Epoch 160/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9201 - loss: 0.1907 - val_accuracy: 0.9475 - val_loss: 0.1557 - learning_rate: 1.0000e-04
    Epoch 161/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.9262 - loss: 0.2019 - val_accuracy: 0.9443 - val_loss: 0.1554 - learning_rate: 1.0000e-04
    Epoch 162/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9149 - loss: 0.2030 - val_accuracy: 0.9464 - val_loss: 0.1547 - learning_rate: 1.0000e-04
    Epoch 163/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.9130 - loss: 0.1967 - val_accuracy: 0.9453 - val_loss: 0.1556 - learning_rate: 1.0000e-04
    Epoch 164/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.9204 - loss: 0.1982 - val_accuracy: 0.9453 - val_loss: 0.1555 - learning_rate: 1.0000e-04
    Epoch 165/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9286 - loss: 0.1903 - val_accuracy: 0.9443 - val_loss: 0.1550 - learning_rate: 1.0000e-04
    Epoch 166/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9183 - loss: 0.1950 - val_accuracy: 0.9453 - val_loss: 0.1555 - learning_rate: 1.0000e-04
    Epoch 167/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9303 - loss: 0.1849 - val_accuracy: 0.9464 - val_loss: 0.1547 - learning_rate: 1.0000e-04
    Epoch 168/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.9240 - loss: 0.2035 - val_accuracy: 0.9432 - val_loss: 0.1562 - learning_rate: 1.0000e-04
    Epoch 169/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.9099 - loss: 0.1958 - val_accuracy: 0.9421 - val_loss: 0.1550 - learning_rate: 1.0000e-04
    Epoch 170/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.9170 - loss: 0.2066 - val_accuracy: 0.9432 - val_loss: 0.1562 - learning_rate: 1.0000e-04
    Epoch 171/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9183 - loss: 0.2022 - val_accuracy: 0.9443 - val_loss: 0.1555 - learning_rate: 1.0000e-04
    Epoch 172/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9218 - loss: 0.1989 - val_accuracy: 0.9411 - val_loss: 0.1559 - learning_rate: 1.0000e-04
    Epoch 173/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9183 - loss: 0.1931 - val_accuracy: 0.9443 - val_loss: 0.1560 - learning_rate: 1.0000e-04
    Epoch 174/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9179 - loss: 0.2098 - val_accuracy: 0.9432 - val_loss: 0.1556 - learning_rate: 1.0000e-04
    Epoch 175/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9236 - loss: 0.1935 - val_accuracy: 0.9443 - val_loss: 0.1554 - learning_rate: 1.0000e-04
    Epoch 176/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9256 - loss: 0.1927 - val_accuracy: 0.9443 - val_loss: 0.1554 - learning_rate: 1.0000e-04
    Epoch 177/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9195 - loss: 0.1993 - val_accuracy: 0.9443 - val_loss: 0.1560 - learning_rate: 1.0000e-04
    Epoch 178/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9179 - loss: 0.1950 - val_accuracy: 0.9443 - val_loss: 0.1570 - learning_rate: 1.0000e-04
    Epoch 179/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9221 - loss: 0.2092 - val_accuracy: 0.9443 - val_loss: 0.1568 - learning_rate: 1.0000e-04
    Epoch 180/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9144 - loss: 0.2207 - val_accuracy: 0.9443 - val_loss: 0.1564 - learning_rate: 1.0000e-04
    Epoch 181/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9124 - loss: 0.2072 - val_accuracy: 0.9432 - val_loss: 0.1557 - learning_rate: 1.0000e-04
    Epoch 182/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.9248 - loss: 0.1844 - val_accuracy: 0.9411 - val_loss: 0.1550 - learning_rate: 1.0000e-04
    Epoch 183/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9193 - loss: 0.1933 - val_accuracy: 0.9453 - val_loss: 0.1554 - learning_rate: 1.0000e-04
    Epoch 184/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9207 - loss: 0.1991 - val_accuracy: 0.9453 - val_loss: 0.1553 - learning_rate: 1.0000e-04
    Epoch 185/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.9228 - loss: 0.1920 - val_accuracy: 0.9443 - val_loss: 0.1570 - learning_rate: 1.0000e-04
    Epoch 186/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9199 - loss: 0.1998 - val_accuracy: 0.9453 - val_loss: 0.1560 - learning_rate: 1.0000e-04
    Epoch 187/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.9183 - loss: 0.1993 - val_accuracy: 0.9443 - val_loss: 0.1558 - learning_rate: 1.0000e-04
    Epoch 188/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9192 - loss: 0.2025 - val_accuracy: 0.9464 - val_loss: 0.1547 - learning_rate: 1.0000e-04
    Epoch 189/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9210 - loss: 0.1974 - val_accuracy: 0.9464 - val_loss: 0.1553 - learning_rate: 1.0000e-04
    Epoch 190/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9156 - loss: 0.2052 - val_accuracy: 0.9464 - val_loss: 0.1552 - learning_rate: 1.0000e-04
    Epoch 191/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9194 - loss: 0.1851 - val_accuracy: 0.9389 - val_loss: 0.1552 - learning_rate: 1.0000e-04
    Epoch 192/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9177 - loss: 0.1998 - val_accuracy: 0.9453 - val_loss: 0.1548 - learning_rate: 1.0000e-04
    Epoch 193/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.9187 - loss: 0.1984 - val_accuracy: 0.9432 - val_loss: 0.1547 - learning_rate: 1.0000e-04
    Epoch 194/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9247 - loss: 0.1857 - val_accuracy: 0.9443 - val_loss: 0.1559 - learning_rate: 1.0000e-04
    Epoch 195/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.9200 - loss: 0.2033 - val_accuracy: 0.9443 - val_loss: 0.1557 - learning_rate: 1.0000e-04
    Epoch 196/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9240 - loss: 0.1857 - val_accuracy: 0.9421 - val_loss: 0.1540 - learning_rate: 1.0000e-04
    Epoch 197/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.9235 - loss: 0.1902 - val_accuracy: 0.9432 - val_loss: 0.1544 - learning_rate: 1.0000e-04
    Epoch 198/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9126 - loss: 0.1977 - val_accuracy: 0.9443 - val_loss: 0.1547 - learning_rate: 1.0000e-04
    Epoch 199/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.9152 - loss: 0.2214 - val_accuracy: 0.9421 - val_loss: 0.1539 - learning_rate: 1.0000e-04
    Epoch 200/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9353 - loss: 0.1871 - val_accuracy: 0.9443 - val_loss: 0.1540 - learning_rate: 1.0000e-04
    Epoch 201/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.9179 - loss: 0.1936 - val_accuracy: 0.9443 - val_loss: 0.1535 - learning_rate: 1.0000e-04
    Epoch 202/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9300 - loss: 0.1852 - val_accuracy: 0.9453 - val_loss: 0.1550 - learning_rate: 1.0000e-04
    Epoch 203/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9239 - loss: 0.1978 - val_accuracy: 0.9453 - val_loss: 0.1556 - learning_rate: 1.0000e-04
    Epoch 204/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9092 - loss: 0.2089 - val_accuracy: 0.9453 - val_loss: 0.1556 - learning_rate: 1.0000e-04
    Epoch 205/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9197 - loss: 0.2034 - val_accuracy: 0.9432 - val_loss: 0.1556 - learning_rate: 1.0000e-04
    Epoch 206/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9230 - loss: 0.1983 - val_accuracy: 0.9453 - val_loss: 0.1541 - learning_rate: 1.0000e-04
    Epoch 207/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.9280 - loss: 0.1871 - val_accuracy: 0.9453 - val_loss: 0.1540 - learning_rate: 1.0000e-04
    Epoch 208/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.9266 - loss: 0.1780 - val_accuracy: 0.9464 - val_loss: 0.1540 - learning_rate: 1.0000e-04
    Epoch 209/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9156 - loss: 0.2090 - val_accuracy: 0.9464 - val_loss: 0.1538 - learning_rate: 1.0000e-04
    Epoch 210/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.9280 - loss: 0.1931 - val_accuracy: 0.9432 - val_loss: 0.1568 - learning_rate: 1.0000e-04
    Epoch 211/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9183 - loss: 0.1914 - val_accuracy: 0.9443 - val_loss: 0.1552 - learning_rate: 1.0000e-04
    Epoch 212/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9287 - loss: 0.1788 - val_accuracy: 0.9443 - val_loss: 0.1553 - learning_rate: 1.0000e-04
    Epoch 213/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.9238 - loss: 0.1844 - val_accuracy: 0.9443 - val_loss: 0.1548 - learning_rate: 1.0000e-04
    Epoch 214/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9227 - loss: 0.1843 - val_accuracy: 0.9443 - val_loss: 0.1549 - learning_rate: 1.0000e-04
    Epoch 215/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9268 - loss: 0.1873 - val_accuracy: 0.9411 - val_loss: 0.1581 - learning_rate: 1.0000e-04
    Epoch 216/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9197 - loss: 0.1970 - val_accuracy: 0.9453 - val_loss: 0.1551 - learning_rate: 1.0000e-04
    Epoch 217/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9203 - loss: 0.1912 - val_accuracy: 0.9453 - val_loss: 0.1538 - learning_rate: 1.0000e-04
    Epoch 218/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9174 - loss: 0.1911 - val_accuracy: 0.9400 - val_loss: 0.1536 - learning_rate: 1.0000e-04
    Epoch 219/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9263 - loss: 0.1977 - val_accuracy: 0.9453 - val_loss: 0.1548 - learning_rate: 1.0000e-04
    Epoch 220/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9232 - loss: 0.1916 - val_accuracy: 0.9464 - val_loss: 0.1530 - learning_rate: 1.0000e-04
    Epoch 221/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9294 - loss: 0.1822 - val_accuracy: 0.9464 - val_loss: 0.1531 - learning_rate: 1.0000e-04
    Epoch 222/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9238 - loss: 0.1974 - val_accuracy: 0.9464 - val_loss: 0.1529 - learning_rate: 1.0000e-04
    Epoch 223/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9223 - loss: 0.1927 - val_accuracy: 0.9453 - val_loss: 0.1540 - learning_rate: 1.0000e-04
    Epoch 224/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9148 - loss: 0.2011 - val_accuracy: 0.9464 - val_loss: 0.1532 - learning_rate: 1.0000e-04
    Epoch 225/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9128 - loss: 0.1980 - val_accuracy: 0.9453 - val_loss: 0.1538 - learning_rate: 1.0000e-04
    Epoch 226/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 35ms/step - accuracy: 0.9211 - loss: 0.1953 - val_accuracy: 0.9453 - val_loss: 0.1522 - learning_rate: 1.0000e-04
    Epoch 227/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.9218 - loss: 0.1932 - val_accuracy: 0.9453 - val_loss: 0.1532 - learning_rate: 1.0000e-04
    Epoch 228/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9200 - loss: 0.2027 - val_accuracy: 0.9453 - val_loss: 0.1537 - learning_rate: 1.0000e-04
    Epoch 229/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9201 - loss: 0.1949 - val_accuracy: 0.9453 - val_loss: 0.1526 - learning_rate: 1.0000e-04
    Epoch 230/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9315 - loss: 0.1914 - val_accuracy: 0.9443 - val_loss: 0.1530 - learning_rate: 1.0000e-04
    Epoch 231/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9224 - loss: 0.1841 - val_accuracy: 0.9464 - val_loss: 0.1536 - learning_rate: 1.0000e-04
    Epoch 232/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.9151 - loss: 0.1994 - val_accuracy: 0.9453 - val_loss: 0.1537 - learning_rate: 1.0000e-04
    Epoch 233/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.9187 - loss: 0.1981 - val_accuracy: 0.9453 - val_loss: 0.1549 - learning_rate: 1.0000e-04
    Epoch 234/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9211 - loss: 0.2006 - val_accuracy: 0.9443 - val_loss: 0.1554 - learning_rate: 1.0000e-04
    Epoch 235/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.9140 - loss: 0.1944 - val_accuracy: 0.9453 - val_loss: 0.1541 - learning_rate: 1.0000e-04
    Epoch 236/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9230 - loss: 0.1957 - val_accuracy: 0.9464 - val_loss: 0.1535 - learning_rate: 1.0000e-04
    Epoch 237/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9273 - loss: 0.1906 - val_accuracy: 0.9443 - val_loss: 0.1534 - learning_rate: 1.0000e-04
    Epoch 238/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9213 - loss: 0.1910 - val_accuracy: 0.9443 - val_loss: 0.1543 - learning_rate: 1.0000e-04
    Epoch 239/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9167 - loss: 0.2033 - val_accuracy: 0.9443 - val_loss: 0.1546 - learning_rate: 1.0000e-04
    Epoch 240/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9318 - loss: 0.1785 - val_accuracy: 0.9453 - val_loss: 0.1553 - learning_rate: 1.0000e-04
    Epoch 241/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9181 - loss: 0.2020 - val_accuracy: 0.9443 - val_loss: 0.1551 - learning_rate: 1.0000e-04
    Epoch 242/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9214 - loss: 0.1843 - val_accuracy: 0.9443 - val_loss: 0.1557 - learning_rate: 1.0000e-04
    Epoch 243/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9153 - loss: 0.2000 - val_accuracy: 0.9453 - val_loss: 0.1537 - learning_rate: 1.0000e-04
    Epoch 244/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9301 - loss: 0.1786 - val_accuracy: 0.9453 - val_loss: 0.1531 - learning_rate: 1.0000e-04
    Epoch 245/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9329 - loss: 0.1682 - val_accuracy: 0.9400 - val_loss: 0.1538 - learning_rate: 1.0000e-04
    Epoch 246/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.9168 - loss: 0.1966 - val_accuracy: 0.9453 - val_loss: 0.1546 - learning_rate: 1.0000e-04
    Epoch 247/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9224 - loss: 0.1840 - val_accuracy: 0.9464 - val_loss: 0.1542 - learning_rate: 1.0000e-04
    Epoch 248/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9233 - loss: 0.1908 - val_accuracy: 0.9443 - val_loss: 0.1535 - learning_rate: 1.0000e-04
    Epoch 249/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9098 - loss: 0.1958 - val_accuracy: 0.9411 - val_loss: 0.1553 - learning_rate: 1.0000e-04
    Epoch 250/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9258 - loss: 0.1918 - val_accuracy: 0.9453 - val_loss: 0.1531 - learning_rate: 1.0000e-04
    Epoch 251/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9232 - loss: 0.1855 - val_accuracy: 0.9432 - val_loss: 0.1539 - learning_rate: 1.0000e-04
    Epoch 252/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9136 - loss: 0.2023 - val_accuracy: 0.9443 - val_loss: 0.1541 - learning_rate: 1.0000e-04
    Epoch 253/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9251 - loss: 0.1823 - val_accuracy: 0.9443 - val_loss: 0.1534 - learning_rate: 1.0000e-04
    Epoch 254/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 35ms/step - accuracy: 0.9200 - loss: 0.1925 - val_accuracy: 0.9453 - val_loss: 0.1536 - learning_rate: 1.0000e-04
    Epoch 255/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 35ms/step - accuracy: 0.9223 - loss: 0.1892 - val_accuracy: 0.9432 - val_loss: 0.1536 - learning_rate: 1.0000e-04
    Epoch 256/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9240 - loss: 0.1859 - val_accuracy: 0.9443 - val_loss: 0.1531 - learning_rate: 1.0000e-04
    Epoch 257/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 35ms/step - accuracy: 0.9182 - loss: 0.1976 - val_accuracy: 0.9411 - val_loss: 0.1568 - learning_rate: 1.0000e-04
    Epoch 258/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9241 - loss: 0.1955 - val_accuracy: 0.9443 - val_loss: 0.1525 - learning_rate: 1.0000e-04
    Epoch 259/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9227 - loss: 0.1935 - val_accuracy: 0.9443 - val_loss: 0.1520 - learning_rate: 1.0000e-04
    Epoch 260/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.9243 - loss: 0.1878 - val_accuracy: 0.9443 - val_loss: 0.1546 - learning_rate: 1.0000e-04
    Epoch 261/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9211 - loss: 0.1880 - val_accuracy: 0.9443 - val_loss: 0.1532 - learning_rate: 1.0000e-04
    Epoch 262/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9198 - loss: 0.2032 - val_accuracy: 0.9453 - val_loss: 0.1533 - learning_rate: 1.0000e-04
    Epoch 263/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9184 - loss: 0.2010 - val_accuracy: 0.9453 - val_loss: 0.1538 - learning_rate: 1.0000e-04
    Epoch 264/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9236 - loss: 0.1811 - val_accuracy: 0.9453 - val_loss: 0.1544 - learning_rate: 1.0000e-04
    Epoch 265/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9229 - loss: 0.1932 - val_accuracy: 0.9453 - val_loss: 0.1530 - learning_rate: 1.0000e-04
    Epoch 266/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9228 - loss: 0.1971 - val_accuracy: 0.9453 - val_loss: 0.1537 - learning_rate: 1.0000e-04
    Epoch 267/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9155 - loss: 0.1997 - val_accuracy: 0.9453 - val_loss: 0.1541 - learning_rate: 1.0000e-04
    Epoch 268/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9291 - loss: 0.1829 - val_accuracy: 0.9453 - val_loss: 0.1539 - learning_rate: 1.0000e-04
    Epoch 269/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9182 - loss: 0.1951 - val_accuracy: 0.9453 - val_loss: 0.1542 - learning_rate: 1.0000e-04
    Epoch 270/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.9207 - loss: 0.1916 - val_accuracy: 0.9453 - val_loss: 0.1540 - learning_rate: 1.0000e-04
    Epoch 271/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9199 - loss: 0.2000 - val_accuracy: 0.9453 - val_loss: 0.1542 - learning_rate: 1.0000e-04
    Epoch 272/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9162 - loss: 0.1899 - val_accuracy: 0.9464 - val_loss: 0.1528 - learning_rate: 1.0000e-04
    Epoch 273/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9271 - loss: 0.1780 - val_accuracy: 0.9443 - val_loss: 0.1531 - learning_rate: 1.0000e-04
    Epoch 274/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9191 - loss: 0.2044 - val_accuracy: 0.9443 - val_loss: 0.1542 - learning_rate: 1.0000e-04
    Epoch 275/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.9215 - loss: 0.1945 - val_accuracy: 0.9464 - val_loss: 0.1522 - learning_rate: 1.0000e-04
    Epoch 276/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9211 - loss: 0.1983 - val_accuracy: 0.9453 - val_loss: 0.1530 - learning_rate: 1.0000e-04
    Epoch 277/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9154 - loss: 0.2079 - val_accuracy: 0.9453 - val_loss: 0.1529 - learning_rate: 1.0000e-04
    Epoch 278/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9193 - loss: 0.1954 - val_accuracy: 0.9453 - val_loss: 0.1523 - learning_rate: 1.0000e-04
    Epoch 279/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9270 - loss: 0.1859 - val_accuracy: 0.9453 - val_loss: 0.1524 - learning_rate: 1.0000e-04
    Epoch 280/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9261 - loss: 0.1804 - val_accuracy: 0.9453 - val_loss: 0.1534 - learning_rate: 1.0000e-04
    Epoch 281/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9275 - loss: 0.1824 - val_accuracy: 0.9453 - val_loss: 0.1525 - learning_rate: 1.0000e-04
    Epoch 282/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 35ms/step - accuracy: 0.9241 - loss: 0.1861 - val_accuracy: 0.9453 - val_loss: 0.1528 - learning_rate: 1.0000e-04
    Epoch 283/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 35ms/step - accuracy: 0.9265 - loss: 0.1910 - val_accuracy: 0.9453 - val_loss: 0.1521 - learning_rate: 1.0000e-04
    Epoch 284/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9284 - loss: 0.1790 - val_accuracy: 0.9453 - val_loss: 0.1529 - learning_rate: 1.0000e-04
    Epoch 285/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9208 - loss: 0.1961 - val_accuracy: 0.9443 - val_loss: 0.1555 - learning_rate: 1.0000e-04
    Epoch 286/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9234 - loss: 0.1964 - val_accuracy: 0.9464 - val_loss: 0.1537 - learning_rate: 1.0000e-04
    Epoch 287/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9217 - loss: 0.1863 - val_accuracy: 0.9443 - val_loss: 0.1568 - learning_rate: 1.0000e-04
    Epoch 288/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9292 - loss: 0.1801 - val_accuracy: 0.9464 - val_loss: 0.1541 - learning_rate: 1.0000e-04
    Epoch 289/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9317 - loss: 0.1917 - val_accuracy: 0.9453 - val_loss: 0.1534 - learning_rate: 1.0000e-04
    Epoch 290/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 32ms/step - accuracy: 0.9222 - loss: 0.1916 - val_accuracy: 0.9464 - val_loss: 0.1526 - learning_rate: 1.0000e-04
    Epoch 291/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9209 - loss: 0.2012 - val_accuracy: 0.9443 - val_loss: 0.1555 - learning_rate: 1.0000e-04
    Epoch 292/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 33ms/step - accuracy: 0.9198 - loss: 0.1974 - val_accuracy: 0.9443 - val_loss: 0.1544 - learning_rate: 1.0000e-04
    Epoch 293/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.9289 - loss: 0.1891 - val_accuracy: 0.9453 - val_loss: 0.1545 - learning_rate: 1.0000e-04
    Epoch 294/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 33ms/step - accuracy: 0.9305 - loss: 0.1902 - val_accuracy: 0.9443 - val_loss: 0.1537 - learning_rate: 1.0000e-04
    Epoch 295/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9186 - loss: 0.1976 - val_accuracy: 0.9443 - val_loss: 0.1537 - learning_rate: 1.0000e-04
    Epoch 296/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9203 - loss: 0.1922 - val_accuracy: 0.9464 - val_loss: 0.1523 - learning_rate: 1.0000e-04
    Epoch 297/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 35ms/step - accuracy: 0.9233 - loss: 0.1924 - val_accuracy: 0.9443 - val_loss: 0.1532 - learning_rate: 1.0000e-04
    Epoch 298/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 35ms/step - accuracy: 0.9255 - loss: 0.1944 - val_accuracy: 0.9443 - val_loss: 0.1535 - learning_rate: 1.0000e-04
    Epoch 299/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 34ms/step - accuracy: 0.9284 - loss: 0.1865 - val_accuracy: 0.9443 - val_loss: 0.1542 - learning_rate: 1.0000e-04
    Epoch 299: early stopping
    Restoring model weights from the end of the best epoch: 259.



```python
# Create dataframe of model fit history.
mod_cnn_hist_df = pd.DataFrame().from_dict(mod_cnn_hist.history, orient = 'columns')

plot_TF_training_history(mod_cnn_hist_df, 'CNN - Training History')
```


    
![png](Periocular_Sex_Classification_files/Periocular_Sex_Classification_82_0.png)
    


This looks great, we can see how the model stabilizes and converges nicely without overfitting, still gaining slight improvements all the way until epoch 259.  During model building and tuning it was unclear if batch normalization layers were necessary in this network, but in the end it didn't seem necessary.

Looking at the training loss line we can see how much the loss oscillates until around 100 epochs this is likely due to the random image augmentations, helping create a more generalizable and robust model.


```python
y_pred_val_cnn_proba = mod_cnn.predict(np.stack(X_val.norm_flat.apply(flat_to_array)), verbose = "auto", callbacks = None)
y_pred_val_cnn = y_pred_val_cnn_proba.argmax(axis = 1)

accuracy_score(y_val, y_pred_val_cnn)
```

    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 17ms/step





    0.9442658092175777



The model achieves a 94.43% accuracy score on the validation set.


```python
y_pred_cnn_proba = mod_cnn.predict(np.stack(X_test.norm_flat.apply(flat_to_array)), verbose = "auto", callbacks = None)
y_pred_cnn = y_pred_cnn_proba.argmax(axis = 1)
```

    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 17ms/step


##### Non-Augmented Training

Train the same model but use the training set that is not augmented/transformed for comparison at the end. Muted outputs for clarity.


```python
mod_cnn_no_aug = build_cnn()

# Callbacks to use (same as above):
early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 40, verbose = 1, restore_best_weights = True)
reduce_lr_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 5, cooldown = 8, min_lr = 0.0001, verbose = 0)
optimizer_param = tf.keras.optimizers.SGD(learning_rate = 0.01)

mod_cnn_no_aug.compile(loss = loss_fx,
                optimizer = optimizer_param,
                metrics = ['accuracy'])

mod_cnn_no_aug_hist = mod_cnn_no_aug.fit(ds_train,
                           batch_size = BATCH_SIZE,
                           epochs = n_epochs,
                           verbose = 0,
                           callbacks = [early_stop, reduce_lr_plateau],
                           validation_split = 0.0,
                           validation_data = ds_val,
                           shuffle = True,
                           class_weight = None,
                           sample_weight = None,
                           initial_epoch = 0,
                           steps_per_epoch = None,
                           validation_steps = None,
                           validation_batch_size = None,
                           validation_freq = val_freq)

y_pred_val_cnn_no_aug_proba = mod_cnn_no_aug.predict(ds_val, verbose = 0, callbacks = None)
y_pred_val_cnn_no_aug = y_pred_val_cnn_no_aug_proba.argmax(axis = 1)

y_pred_cnn_no_aug_proba = mod_cnn_no_aug.predict(np.stack(X_test.norm_flat.apply(flat_to_array)), verbose = 0, callbacks = None)
y_pred_cnn_no_aug = y_pred_cnn_no_aug_proba.argmax(axis = 1)
```

    Epoch 68: early stopping
    Restoring model weights from the end of the best epoch: 28.


Interestingly the model without augmented training images stops training and reverts back to weights from epoch 28.

###### [Back to Table of Contents](#toc)

#### 7.2.3. Deep Convolutional Neural Network (DCNN): <a name="dcnn"></a>

Next, I would like to try to implement a model that is tried and tested on image classification tasks-ResNet50- which is the perfect mix of depth and efficiency here. Anything deeper would likely be overkill since the images input are only 56x56.

The benefit of the ResNet architecture is within it's name, Residual connections, sometimes called skip connections. These connections connect/concatenate outputs of earlier layers to inputs of later layers, skipping over a number of layers (in this case 2). This effectively combats the issue of vanishing gradients by allowing the network to skip over connections that hurt the performance of the classifier.

This model below was largely built from the source code for TensorFlow's ResNet50 from https://github.com/tensorflow/models/tree/master/official/vision and elsewhere. Slight alterations were made to the original architecture but it is mostly the same. Notably, I opted for a smaller kernel size in the first layer as this dataset's images are already very small. There are pre-trained versions of ResNet out there (and within Tensorflow for that matter), but I will build and train a new one for a more equal comparison between this and my previous, much shallower model.


```python
# Inspired by ResNet 50 https://github.com/tensorflow/models/tree/master/official/vision
# Residual function to set skip connections. Downsample to reduce dimension.
def residual_block(x, filters, downsample = False):
    shortcut = x
    strides = (2, 2) if downsample else (1, 1)

    # First convolution layer.
    x = tf.keras.layers.Conv2D(filters, kernel_size = (3, 3), strides = strides, padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Second convolution layer.
    x = tf.keras.layers.Conv2D(filters, kernel_size = (3, 3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    #  Add convolution if downsample == True.
    if downsample:
        shortcut = tf.keras.layers.Conv2D(filters, kernel_size = (1, 1), strides = (2, 2), padding = 'same')(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    # Add the shortcut (input) to the output.
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)

    return x

# Build model.
def build_dcnn(input_shape = (img_size, img_size, 3)):
    inputs = tf.keras.Input(shape = input_shape)

    # Initial convolutional layer. Reduced kernel size here from default 7,7 because the input images are small.
    x = tf.keras.layers.Conv2D(64, kernel_size = (5, 5), strides = (2, 2), padding = 'same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)

    # First residual block (3 blocks).
    for _ in range(3):
        x = residual_block(x, 64)

    # Second residual block (4 blocks).
    x = residual_block(x, 128, downsample = True)
    for _ in range(3):
        x = residual_block(x, 128)

    # Third residual block (6 blocks).
    x = residual_block(x, 256, downsample = True)
    for _ in range(5):
        x = residual_block(x, 256)

    # Fourth residual block (3 blocks).
    x = residual_block(x, 512, downsample = True)
    for _ in range(2):
        x = residual_block(x, 512)

    # Global average pooling layer and final output prediction.
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(2, name = 'Predictions')(x)

    # Create model.
    model = tf.keras.Model(inputs, outputs)
    return model

# Finally, build the model.
mod_dcnn = build_dcnn(input_shape=(img_size, img_size, 3))
mod_dcnn.summary()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "functional_5"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)        </span>┃<span style="font-weight: bold"> Output Shape      </span>┃<span style="font-weight: bold">    Param # </span>┃<span style="font-weight: bold"> Connected to      </span>┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ input_layer_1       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ -                 │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_10 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>,    │      <span style="color: #00af00; text-decoration-color: #00af00">4,864</span> │ input_layer_1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalization │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>,    │        <span style="color: #00af00; text-decoration-color: #00af00">256</span> │ conv2d_10[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ re_lu_18 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling2d_4     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ re_lu_18[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)      │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_11 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>,    │     <span style="color: #00af00; text-decoration-color: #00af00">36,928</span> │ max_pooling2d_4[<span style="color: #00af00; text-decoration-color: #00af00">…</span> │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>,    │        <span style="color: #00af00; text-decoration-color: #00af00">256</span> │ conv2d_11[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ re_lu_19 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_12 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>,    │     <span style="color: #00af00; text-decoration-color: #00af00">36,928</span> │ re_lu_19[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>,    │        <span style="color: #00af00; text-decoration-color: #00af00">256</span> │ conv2d_12[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │ max_pooling2d_4[<span style="color: #00af00; text-decoration-color: #00af00">…</span> │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ re_lu_20 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ add[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]         │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_13 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>,    │     <span style="color: #00af00; text-decoration-color: #00af00">36,928</span> │ re_lu_20[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>,    │        <span style="color: #00af00; text-decoration-color: #00af00">256</span> │ conv2d_13[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ re_lu_21 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_14 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>,    │     <span style="color: #00af00; text-decoration-color: #00af00">36,928</span> │ re_lu_21[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>,    │        <span style="color: #00af00; text-decoration-color: #00af00">256</span> │ conv2d_14[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │ re_lu_20[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ re_lu_22 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ add_1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]       │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_15 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>,    │     <span style="color: #00af00; text-decoration-color: #00af00">36,928</span> │ re_lu_22[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>,    │        <span style="color: #00af00; text-decoration-color: #00af00">256</span> │ conv2d_15[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ re_lu_23 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_16 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>,    │     <span style="color: #00af00; text-decoration-color: #00af00">36,928</span> │ re_lu_23[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>,    │        <span style="color: #00af00; text-decoration-color: #00af00">256</span> │ conv2d_16[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │ re_lu_22[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ re_lu_24 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ add_2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]       │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_17 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>) │     <span style="color: #00af00; text-decoration-color: #00af00">73,856</span> │ re_lu_24[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>) │        <span style="color: #00af00; text-decoration-color: #00af00">512</span> │ conv2d_17[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ re_lu_25 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_18 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>) │    <span style="color: #00af00; text-decoration-color: #00af00">147,584</span> │ re_lu_25[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_19 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>) │      <span style="color: #00af00; text-decoration-color: #00af00">8,320</span> │ re_lu_24[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>) │        <span style="color: #00af00; text-decoration-color: #00af00">512</span> │ conv2d_18[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>) │        <span style="color: #00af00; text-decoration-color: #00af00">512</span> │ conv2d_19[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│                     │                   │            │ batch_normalizat… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ re_lu_26 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ add_3[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]       │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_20 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>) │    <span style="color: #00af00; text-decoration-color: #00af00">147,584</span> │ re_lu_26[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>) │        <span style="color: #00af00; text-decoration-color: #00af00">512</span> │ conv2d_20[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ re_lu_27 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_21 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>) │    <span style="color: #00af00; text-decoration-color: #00af00">147,584</span> │ re_lu_27[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>) │        <span style="color: #00af00; text-decoration-color: #00af00">512</span> │ conv2d_21[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│                     │                   │            │ re_lu_26[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ re_lu_28 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ add_4[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]       │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_22 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>) │    <span style="color: #00af00; text-decoration-color: #00af00">147,584</span> │ re_lu_28[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>) │        <span style="color: #00af00; text-decoration-color: #00af00">512</span> │ conv2d_22[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ re_lu_29 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_23 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>) │    <span style="color: #00af00; text-decoration-color: #00af00">147,584</span> │ re_lu_29[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>) │        <span style="color: #00af00; text-decoration-color: #00af00">512</span> │ conv2d_23[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_5 (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│                     │                   │            │ re_lu_28[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ re_lu_30 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ add_5[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]       │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_24 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>) │    <span style="color: #00af00; text-decoration-color: #00af00">147,584</span> │ re_lu_30[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>) │        <span style="color: #00af00; text-decoration-color: #00af00">512</span> │ conv2d_24[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ re_lu_31 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_25 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>) │    <span style="color: #00af00; text-decoration-color: #00af00">147,584</span> │ re_lu_31[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>) │        <span style="color: #00af00; text-decoration-color: #00af00">512</span> │ conv2d_25[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_6 (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│                     │                   │            │ re_lu_30[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ re_lu_32 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ add_6[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]       │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_26 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │    <span style="color: #00af00; text-decoration-color: #00af00">295,168</span> │ re_lu_32[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │      <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ conv2d_26[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ re_lu_33 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_27 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │    <span style="color: #00af00; text-decoration-color: #00af00">590,080</span> │ re_lu_33[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_28 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │     <span style="color: #00af00; text-decoration-color: #00af00">33,024</span> │ re_lu_32[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │      <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ conv2d_27[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │      <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ conv2d_28[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_7 (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│                     │                   │            │ batch_normalizat… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ re_lu_34 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ add_7[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]       │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_29 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │    <span style="color: #00af00; text-decoration-color: #00af00">590,080</span> │ re_lu_34[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │      <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ conv2d_29[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ re_lu_35 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_30 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │    <span style="color: #00af00; text-decoration-color: #00af00">590,080</span> │ re_lu_35[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │      <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ conv2d_30[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_8 (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│                     │                   │            │ re_lu_34[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ re_lu_36 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ add_8[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]       │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_31 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │    <span style="color: #00af00; text-decoration-color: #00af00">590,080</span> │ re_lu_36[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │      <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ conv2d_31[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ re_lu_37 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_32 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │    <span style="color: #00af00; text-decoration-color: #00af00">590,080</span> │ re_lu_37[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │      <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ conv2d_32[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_9 (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│                     │                   │            │ re_lu_36[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ re_lu_38 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ add_9[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]       │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_33 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │    <span style="color: #00af00; text-decoration-color: #00af00">590,080</span> │ re_lu_38[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │      <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ conv2d_33[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ re_lu_39 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_34 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │    <span style="color: #00af00; text-decoration-color: #00af00">590,080</span> │ re_lu_39[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │      <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ conv2d_34[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_10 (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│                     │                   │            │ re_lu_38[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ re_lu_40 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ add_10[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_35 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │    <span style="color: #00af00; text-decoration-color: #00af00">590,080</span> │ re_lu_40[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │      <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ conv2d_35[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ re_lu_41 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_36 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │    <span style="color: #00af00; text-decoration-color: #00af00">590,080</span> │ re_lu_41[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │      <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ conv2d_36[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_11 (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│                     │                   │            │ re_lu_40[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ re_lu_42 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ add_11[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_37 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │    <span style="color: #00af00; text-decoration-color: #00af00">590,080</span> │ re_lu_42[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │      <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ conv2d_37[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ re_lu_43 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_38 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │    <span style="color: #00af00; text-decoration-color: #00af00">590,080</span> │ re_lu_43[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │      <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ conv2d_38[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_12 (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│                     │                   │            │ re_lu_42[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ re_lu_44 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ add_12[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_39 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │  <span style="color: #00af00; text-decoration-color: #00af00">1,180,160</span> │ re_lu_44[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │      <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv2d_39[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ re_lu_45 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_40 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │  <span style="color: #00af00; text-decoration-color: #00af00">2,359,808</span> │ re_lu_45[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_41 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │    <span style="color: #00af00; text-decoration-color: #00af00">131,584</span> │ re_lu_44[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │      <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv2d_40[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │      <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv2d_41[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_13 (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│                     │                   │            │ batch_normalizat… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ re_lu_46 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ add_13[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_42 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │  <span style="color: #00af00; text-decoration-color: #00af00">2,359,808</span> │ re_lu_46[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │      <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv2d_42[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ re_lu_47 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_43 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │  <span style="color: #00af00; text-decoration-color: #00af00">2,359,808</span> │ re_lu_47[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │      <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv2d_43[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_14 (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│                     │                   │            │ re_lu_46[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ re_lu_48 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ add_14[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_44 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │  <span style="color: #00af00; text-decoration-color: #00af00">2,359,808</span> │ re_lu_48[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │      <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv2d_44[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ re_lu_49 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_45 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │  <span style="color: #00af00; text-decoration-color: #00af00">2,359,808</span> │ re_lu_49[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │      <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv2d_45[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_15 (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│                     │                   │            │ re_lu_48[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ re_lu_50 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ add_15[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ global_average_poo… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)       │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ re_lu_50[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">GlobalAveragePool…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ Predictions (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>)         │      <span style="color: #00af00; text-decoration-color: #00af00">1,026</span> │ global_average_p… │
└─────────────────────┴───────────────────┴────────────┴───────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">21,306,626</span> (81.28 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">21,289,602</span> (81.21 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">17,024</span> (66.50 KB)
</pre>




```python
visualkeras.layered_view(mod_dcnn, legend = True)
```




    
![png](Periocular_Sex_Classification_files/Periocular_Sex_Classification_92_0.png)
    



As expected, that's a lot of layers. Also as another example of the power of stacked 3x3 layers, note the number of trainable parameters here - 21,289,602. That's almost the same amount as my the previous model's architecture with 5x5 and 7x7 kernels!

As with all other models the same callbacks are used, but were slightly modified to suit this network.


```python
# Using similar loss, optimizer, and callbacks here.
loss_fx = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True) # Integer labels.
optimizer_param = 'sgd'
val_freq = 1
n_epochs = 300
# Callbacks to use:
early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 30, verbose = 1, restore_best_weights = True)
reduce_lr_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', 
                                                         factor = 0.5,
                                                         patience = 5,
                                                         cooldown = 8,
                                                         min_lr = 0.0001,
                                                         verbose = 1)
```


```python
mod_dcnn.compile(loss = loss_fx,
                optimizer = optimizer_param,
                metrics = ['accuracy'])

mod_dcnn_hist = mod_dcnn.fit(ds_train_transformed,
                           batch_size = BATCH_SIZE,
                           epochs = n_epochs,
                           verbose = "auto",
                           callbacks = [early_stop, reduce_lr_plateau],
                           validation_split = 0.0,
                           validation_data = ds_val,
                           shuffle = True,
                           class_weight = None,
                           sample_weight = None,
                           initial_epoch = 0,
                           steps_per_epoch = None,
                           validation_steps = None,
                           validation_batch_size = None,
                           validation_freq = val_freq)
```

    Epoch 1/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m58s[0m 240ms/step - accuracy: 0.5159 - loss: 1.3220 - val_accuracy: 0.6120 - val_loss: 0.6770 - learning_rate: 0.0100
    Epoch 2/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 238ms/step - accuracy: 0.6021 - loss: 0.8050 - val_accuracy: 0.7138 - val_loss: 0.5994 - learning_rate: 0.0100
    Epoch 3/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 234ms/step - accuracy: 0.6490 - loss: 0.6986 - val_accuracy: 0.7353 - val_loss: 0.5611 - learning_rate: 0.0100
    Epoch 4/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 241ms/step - accuracy: 0.7088 - loss: 0.5964 - val_accuracy: 0.7771 - val_loss: 0.4710 - learning_rate: 0.0100
    Epoch 5/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 238ms/step - accuracy: 0.7268 - loss: 0.5578 - val_accuracy: 0.7953 - val_loss: 0.4342 - learning_rate: 0.0100
    Epoch 6/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 238ms/step - accuracy: 0.7410 - loss: 0.5364 - val_accuracy: 0.7599 - val_loss: 0.5030 - learning_rate: 0.0100
    Epoch 7/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 240ms/step - accuracy: 0.7577 - loss: 0.4890 - val_accuracy: 0.8124 - val_loss: 0.4142 - learning_rate: 0.0100
    Epoch 8/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 234ms/step - accuracy: 0.7613 - loss: 0.4814 - val_accuracy: 0.7320 - val_loss: 0.5295 - learning_rate: 0.0100
    Epoch 9/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 234ms/step - accuracy: 0.7802 - loss: 0.4428 - val_accuracy: 0.8049 - val_loss: 0.4384 - learning_rate: 0.0100
    Epoch 10/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 234ms/step - accuracy: 0.8045 - loss: 0.4365 - val_accuracy: 0.8435 - val_loss: 0.3543 - learning_rate: 0.0100
    Epoch 11/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 242ms/step - accuracy: 0.8004 - loss: 0.4236 - val_accuracy: 0.8703 - val_loss: 0.3057 - learning_rate: 0.0100
    Epoch 12/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 245ms/step - accuracy: 0.8095 - loss: 0.4275 - val_accuracy: 0.7406 - val_loss: 0.5253 - learning_rate: 0.0100
    Epoch 13/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 242ms/step - accuracy: 0.8204 - loss: 0.4064 - val_accuracy: 0.8735 - val_loss: 0.2879 - learning_rate: 0.0100
    Epoch 14/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 240ms/step - accuracy: 0.8389 - loss: 0.3760 - val_accuracy: 0.8725 - val_loss: 0.3099 - learning_rate: 0.0100
    Epoch 15/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 242ms/step - accuracy: 0.8382 - loss: 0.3619 - val_accuracy: 0.8242 - val_loss: 0.4151 - learning_rate: 0.0100
    Epoch 16/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 242ms/step - accuracy: 0.8570 - loss: 0.3319 - val_accuracy: 0.7513 - val_loss: 0.5177 - learning_rate: 0.0100
    Epoch 17/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 243ms/step - accuracy: 0.8465 - loss: 0.3442 - val_accuracy: 0.8574 - val_loss: 0.3644 - learning_rate: 0.0100
    Epoch 18/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 228ms/step - accuracy: 0.8431 - loss: 0.3518
    Epoch 18: ReduceLROnPlateau reducing learning rate to 0.004999999888241291.
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 240ms/step - accuracy: 0.8432 - loss: 0.3517 - val_accuracy: 0.8135 - val_loss: 0.4336 - learning_rate: 0.0100
    Epoch 19/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 244ms/step - accuracy: 0.8719 - loss: 0.2980 - val_accuracy: 0.8939 - val_loss: 0.2597 - learning_rate: 0.0050
    Epoch 20/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 237ms/step - accuracy: 0.8707 - loss: 0.2879 - val_accuracy: 0.8950 - val_loss: 0.2823 - learning_rate: 0.0050
    Epoch 21/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 236ms/step - accuracy: 0.8799 - loss: 0.2852 - val_accuracy: 0.9164 - val_loss: 0.2090 - learning_rate: 0.0050
    Epoch 22/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 237ms/step - accuracy: 0.8847 - loss: 0.2809 - val_accuracy: 0.9057 - val_loss: 0.2552 - learning_rate: 0.0050
    Epoch 23/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 241ms/step - accuracy: 0.8870 - loss: 0.2738 - val_accuracy: 0.8800 - val_loss: 0.2736 - learning_rate: 0.0050
    Epoch 24/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 242ms/step - accuracy: 0.8944 - loss: 0.2663 - val_accuracy: 0.9025 - val_loss: 0.2290 - learning_rate: 0.0050
    Epoch 25/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 242ms/step - accuracy: 0.8937 - loss: 0.2671 - val_accuracy: 0.8950 - val_loss: 0.2500 - learning_rate: 0.0050
    Epoch 26/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 239ms/step - accuracy: 0.8953 - loss: 0.2642 - val_accuracy: 0.9110 - val_loss: 0.2217 - learning_rate: 0.0050
    Epoch 27/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 240ms/step - accuracy: 0.9016 - loss: 0.2425 - val_accuracy: 0.8992 - val_loss: 0.2435 - learning_rate: 0.0050
    Epoch 28/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 237ms/step - accuracy: 0.8895 - loss: 0.2738 - val_accuracy: 0.9089 - val_loss: 0.2514 - learning_rate: 0.0050
    Epoch 29/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 241ms/step - accuracy: 0.8967 - loss: 0.2497 - val_accuracy: 0.9175 - val_loss: 0.2059 - learning_rate: 0.0050
    Epoch 30/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 239ms/step - accuracy: 0.8842 - loss: 0.2688 - val_accuracy: 0.8585 - val_loss: 0.3477 - learning_rate: 0.0050
    Epoch 31/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 237ms/step - accuracy: 0.9110 - loss: 0.2226 - val_accuracy: 0.9121 - val_loss: 0.2395 - learning_rate: 0.0050
    Epoch 32/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 237ms/step - accuracy: 0.8957 - loss: 0.2386 - val_accuracy: 0.8800 - val_loss: 0.2733 - learning_rate: 0.0050
    Epoch 33/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 240ms/step - accuracy: 0.9070 - loss: 0.2415 - val_accuracy: 0.9164 - val_loss: 0.2105 - learning_rate: 0.0050
    Epoch 34/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 227ms/step - accuracy: 0.9065 - loss: 0.2333
    Epoch 34: ReduceLROnPlateau reducing learning rate to 0.0024999999441206455.
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 239ms/step - accuracy: 0.9064 - loss: 0.2333 - val_accuracy: 0.8296 - val_loss: 0.3811 - learning_rate: 0.0050
    Epoch 35/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 237ms/step - accuracy: 0.9049 - loss: 0.2200 - val_accuracy: 0.9271 - val_loss: 0.1888 - learning_rate: 0.0025
    Epoch 36/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 241ms/step - accuracy: 0.9166 - loss: 0.2086 - val_accuracy: 0.9250 - val_loss: 0.1979 - learning_rate: 0.0025
    Epoch 37/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 244ms/step - accuracy: 0.9098 - loss: 0.2190 - val_accuracy: 0.9078 - val_loss: 0.2182 - learning_rate: 0.0025
    Epoch 38/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 239ms/step - accuracy: 0.9107 - loss: 0.2249 - val_accuracy: 0.9314 - val_loss: 0.1888 - learning_rate: 0.0025
    Epoch 39/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 244ms/step - accuracy: 0.9195 - loss: 0.2015 - val_accuracy: 0.9293 - val_loss: 0.1879 - learning_rate: 0.0025
    Epoch 40/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 237ms/step - accuracy: 0.9189 - loss: 0.2037 - val_accuracy: 0.9303 - val_loss: 0.1782 - learning_rate: 0.0025
    Epoch 41/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 244ms/step - accuracy: 0.9242 - loss: 0.1932 - val_accuracy: 0.9282 - val_loss: 0.1821 - learning_rate: 0.0025
    Epoch 42/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 244ms/step - accuracy: 0.9173 - loss: 0.1997 - val_accuracy: 0.9282 - val_loss: 0.1796 - learning_rate: 0.0025
    Epoch 43/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 243ms/step - accuracy: 0.9232 - loss: 0.1973 - val_accuracy: 0.9293 - val_loss: 0.1809 - learning_rate: 0.0025
    Epoch 44/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 242ms/step - accuracy: 0.9209 - loss: 0.1923 - val_accuracy: 0.8853 - val_loss: 0.3099 - learning_rate: 0.0025
    Epoch 45/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 244ms/step - accuracy: 0.9233 - loss: 0.1947 - val_accuracy: 0.9196 - val_loss: 0.2263 - learning_rate: 0.0025
    Epoch 46/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 244ms/step - accuracy: 0.9084 - loss: 0.2117 - val_accuracy: 0.9346 - val_loss: 0.1766 - learning_rate: 0.0025
    Epoch 47/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 240ms/step - accuracy: 0.9191 - loss: 0.1876 - val_accuracy: 0.9293 - val_loss: 0.1790 - learning_rate: 0.0025
    Epoch 48/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 239ms/step - accuracy: 0.9171 - loss: 0.1813 - val_accuracy: 0.9207 - val_loss: 0.2091 - learning_rate: 0.0025
    Epoch 49/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 240ms/step - accuracy: 0.9186 - loss: 0.1983 - val_accuracy: 0.9368 - val_loss: 0.1694 - learning_rate: 0.0025
    Epoch 50/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 242ms/step - accuracy: 0.9243 - loss: 0.1881 - val_accuracy: 0.9314 - val_loss: 0.1777 - learning_rate: 0.0025
    Epoch 51/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 241ms/step - accuracy: 0.9324 - loss: 0.1726 - val_accuracy: 0.9014 - val_loss: 0.2521 - learning_rate: 0.0025
    Epoch 52/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 239ms/step - accuracy: 0.9193 - loss: 0.1911 - val_accuracy: 0.9335 - val_loss: 0.1771 - learning_rate: 0.0025
    Epoch 53/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 241ms/step - accuracy: 0.9229 - loss: 0.1789 - val_accuracy: 0.9218 - val_loss: 0.2242 - learning_rate: 0.0025
    Epoch 54/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 231ms/step - accuracy: 0.9405 - loss: 0.1582
    Epoch 54: ReduceLROnPlateau reducing learning rate to 0.0012499999720603228.
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 242ms/step - accuracy: 0.9405 - loss: 0.1582 - val_accuracy: 0.9389 - val_loss: 0.1799 - learning_rate: 0.0025
    Epoch 55/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 242ms/step - accuracy: 0.9292 - loss: 0.1820 - val_accuracy: 0.9303 - val_loss: 0.1933 - learning_rate: 0.0012
    Epoch 56/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 240ms/step - accuracy: 0.9353 - loss: 0.1626 - val_accuracy: 0.9357 - val_loss: 0.1613 - learning_rate: 0.0012
    Epoch 57/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 238ms/step - accuracy: 0.9330 - loss: 0.1691 - val_accuracy: 0.9346 - val_loss: 0.1699 - learning_rate: 0.0012
    Epoch 58/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 232ms/step - accuracy: 0.9313 - loss: 0.1723 - val_accuracy: 0.9325 - val_loss: 0.2126 - learning_rate: 0.0012
    Epoch 59/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 234ms/step - accuracy: 0.9368 - loss: 0.1595 - val_accuracy: 0.9421 - val_loss: 0.1676 - learning_rate: 0.0012
    Epoch 60/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 238ms/step - accuracy: 0.9356 - loss: 0.1692 - val_accuracy: 0.9368 - val_loss: 0.1806 - learning_rate: 0.0012
    Epoch 61/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 239ms/step - accuracy: 0.9346 - loss: 0.1639 - val_accuracy: 0.9400 - val_loss: 0.1685 - learning_rate: 0.0012
    Epoch 62/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 235ms/step - accuracy: 0.9293 - loss: 0.1634 - val_accuracy: 0.9357 - val_loss: 0.1912 - learning_rate: 0.0012
    Epoch 63/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m52s[0m 230ms/step - accuracy: 0.9307 - loss: 0.1719 - val_accuracy: 0.9346 - val_loss: 0.1623 - learning_rate: 0.0012
    Epoch 64/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 232ms/step - accuracy: 0.9440 - loss: 0.1510 - val_accuracy: 0.9421 - val_loss: 0.1583 - learning_rate: 0.0012
    Epoch 65/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 240ms/step - accuracy: 0.9388 - loss: 0.1519 - val_accuracy: 0.9368 - val_loss: 0.1832 - learning_rate: 0.0012
    Epoch 66/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 245ms/step - accuracy: 0.9415 - loss: 0.1522 - val_accuracy: 0.9260 - val_loss: 0.2171 - learning_rate: 0.0012
    Epoch 67/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 238ms/step - accuracy: 0.9460 - loss: 0.1427 - val_accuracy: 0.9335 - val_loss: 0.1790 - learning_rate: 0.0012
    Epoch 68/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 237ms/step - accuracy: 0.9333 - loss: 0.1567 - val_accuracy: 0.9411 - val_loss: 0.1793 - learning_rate: 0.0012
    Epoch 69/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 229ms/step - accuracy: 0.9337 - loss: 0.1603
    Epoch 69: ReduceLROnPlateau reducing learning rate to 0.0006249999860301614.
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 241ms/step - accuracy: 0.9338 - loss: 0.1603 - val_accuracy: 0.9346 - val_loss: 0.1919 - learning_rate: 0.0012
    Epoch 70/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 240ms/step - accuracy: 0.9346 - loss: 0.1541 - val_accuracy: 0.9378 - val_loss: 0.1620 - learning_rate: 6.2500e-04
    Epoch 71/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 243ms/step - accuracy: 0.9464 - loss: 0.1414 - val_accuracy: 0.9357 - val_loss: 0.1653 - learning_rate: 6.2500e-04
    Epoch 72/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 241ms/step - accuracy: 0.9308 - loss: 0.1532 - val_accuracy: 0.9368 - val_loss: 0.1639 - learning_rate: 6.2500e-04
    Epoch 73/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 241ms/step - accuracy: 0.9478 - loss: 0.1418 - val_accuracy: 0.9357 - val_loss: 0.1656 - learning_rate: 6.2500e-04
    Epoch 74/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 243ms/step - accuracy: 0.9423 - loss: 0.1571 - val_accuracy: 0.9207 - val_loss: 0.1945 - learning_rate: 6.2500e-04
    Epoch 75/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 245ms/step - accuracy: 0.9487 - loss: 0.1415 - val_accuracy: 0.9389 - val_loss: 0.1745 - learning_rate: 6.2500e-04
    Epoch 76/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 249ms/step - accuracy: 0.9447 - loss: 0.1312 - val_accuracy: 0.9357 - val_loss: 0.1790 - learning_rate: 6.2500e-04
    Epoch 77/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 238ms/step - accuracy: 0.9427 - loss: 0.1393 - val_accuracy: 0.9411 - val_loss: 0.1599 - learning_rate: 6.2500e-04
    Epoch 78/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 245ms/step - accuracy: 0.9524 - loss: 0.1367 - val_accuracy: 0.9293 - val_loss: 0.1662 - learning_rate: 6.2500e-04
    Epoch 79/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 240ms/step - accuracy: 0.9450 - loss: 0.1496 - val_accuracy: 0.9411 - val_loss: 0.1693 - learning_rate: 6.2500e-04
    Epoch 80/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 241ms/step - accuracy: 0.9459 - loss: 0.1295 - val_accuracy: 0.9303 - val_loss: 0.1984 - learning_rate: 6.2500e-04
    Epoch 81/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 232ms/step - accuracy: 0.9487 - loss: 0.1411
    Epoch 81: ReduceLROnPlateau reducing learning rate to 0.0003124999930150807.
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 245ms/step - accuracy: 0.9487 - loss: 0.1411 - val_accuracy: 0.9378 - val_loss: 0.1812 - learning_rate: 6.2500e-04
    Epoch 82/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 246ms/step - accuracy: 0.9483 - loss: 0.1393 - val_accuracy: 0.9411 - val_loss: 0.1701 - learning_rate: 3.1250e-04
    Epoch 83/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 243ms/step - accuracy: 0.9460 - loss: 0.1446 - val_accuracy: 0.9378 - val_loss: 0.1759 - learning_rate: 3.1250e-04
    Epoch 84/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 240ms/step - accuracy: 0.9503 - loss: 0.1326 - val_accuracy: 0.9378 - val_loss: 0.1694 - learning_rate: 3.1250e-04
    Epoch 85/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 241ms/step - accuracy: 0.9454 - loss: 0.1365 - val_accuracy: 0.9389 - val_loss: 0.1686 - learning_rate: 3.1250e-04
    Epoch 86/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 239ms/step - accuracy: 0.9518 - loss: 0.1207 - val_accuracy: 0.9432 - val_loss: 0.1537 - learning_rate: 3.1250e-04
    Epoch 87/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 244ms/step - accuracy: 0.9485 - loss: 0.1308 - val_accuracy: 0.9400 - val_loss: 0.1558 - learning_rate: 3.1250e-04
    Epoch 88/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 243ms/step - accuracy: 0.9413 - loss: 0.1566 - val_accuracy: 0.9443 - val_loss: 0.1584 - learning_rate: 3.1250e-04
    Epoch 89/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 244ms/step - accuracy: 0.9504 - loss: 0.1267 - val_accuracy: 0.9411 - val_loss: 0.1633 - learning_rate: 3.1250e-04
    Epoch 90/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 244ms/step - accuracy: 0.9491 - loss: 0.1356 - val_accuracy: 0.9411 - val_loss: 0.1576 - learning_rate: 3.1250e-04
    Epoch 91/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 247ms/step - accuracy: 0.9436 - loss: 0.1392 - val_accuracy: 0.9432 - val_loss: 0.1540 - learning_rate: 3.1250e-04
    Epoch 92/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 243ms/step - accuracy: 0.9485 - loss: 0.1341 - val_accuracy: 0.9453 - val_loss: 0.1556 - learning_rate: 3.1250e-04
    Epoch 93/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 230ms/step - accuracy: 0.9603 - loss: 0.1175
    Epoch 93: ReduceLROnPlateau reducing learning rate to 0.00015624999650754035.
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 242ms/step - accuracy: 0.9603 - loss: 0.1176 - val_accuracy: 0.9389 - val_loss: 0.1587 - learning_rate: 3.1250e-04
    Epoch 94/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 243ms/step - accuracy: 0.9565 - loss: 0.1160 - val_accuracy: 0.9421 - val_loss: 0.1549 - learning_rate: 1.5625e-04
    Epoch 95/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 242ms/step - accuracy: 0.9471 - loss: 0.1452 - val_accuracy: 0.9421 - val_loss: 0.1574 - learning_rate: 1.5625e-04
    Epoch 96/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 244ms/step - accuracy: 0.9598 - loss: 0.1135 - val_accuracy: 0.9453 - val_loss: 0.1524 - learning_rate: 1.5625e-04
    Epoch 97/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 247ms/step - accuracy: 0.9565 - loss: 0.1260 - val_accuracy: 0.9486 - val_loss: 0.1548 - learning_rate: 1.5625e-04
    Epoch 98/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 250ms/step - accuracy: 0.9582 - loss: 0.1107 - val_accuracy: 0.9378 - val_loss: 0.1660 - learning_rate: 1.5625e-04
    Epoch 99/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 248ms/step - accuracy: 0.9475 - loss: 0.1307 - val_accuracy: 0.9368 - val_loss: 0.1635 - learning_rate: 1.5625e-04
    Epoch 100/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 249ms/step - accuracy: 0.9512 - loss: 0.1254 - val_accuracy: 0.9464 - val_loss: 0.1579 - learning_rate: 1.5625e-04
    Epoch 101/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 246ms/step - accuracy: 0.9481 - loss: 0.1337 - val_accuracy: 0.9453 - val_loss: 0.1571 - learning_rate: 1.5625e-04
    Epoch 102/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 250ms/step - accuracy: 0.9447 - loss: 0.1389 - val_accuracy: 0.9411 - val_loss: 0.1592 - learning_rate: 1.5625e-04
    Epoch 103/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 249ms/step - accuracy: 0.9584 - loss: 0.1148 - val_accuracy: 0.9368 - val_loss: 0.1688 - learning_rate: 1.5625e-04
    Epoch 104/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 246ms/step - accuracy: 0.9482 - loss: 0.1348 - val_accuracy: 0.9464 - val_loss: 0.1584 - learning_rate: 1.5625e-04
    Epoch 105/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 238ms/step - accuracy: 0.9530 - loss: 0.1311
    Epoch 105: ReduceLROnPlateau reducing learning rate to 0.0001.
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 249ms/step - accuracy: 0.9530 - loss: 0.1311 - val_accuracy: 0.9443 - val_loss: 0.1575 - learning_rate: 1.5625e-04
    Epoch 106/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 248ms/step - accuracy: 0.9490 - loss: 0.1325 - val_accuracy: 0.9400 - val_loss: 0.1600 - learning_rate: 1.0000e-04
    Epoch 107/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 247ms/step - accuracy: 0.9428 - loss: 0.1523 - val_accuracy: 0.9411 - val_loss: 0.1599 - learning_rate: 1.0000e-04
    Epoch 108/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 246ms/step - accuracy: 0.9530 - loss: 0.1174 - val_accuracy: 0.9421 - val_loss: 0.1592 - learning_rate: 1.0000e-04
    Epoch 109/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 251ms/step - accuracy: 0.9472 - loss: 0.1252 - val_accuracy: 0.9432 - val_loss: 0.1573 - learning_rate: 1.0000e-04
    Epoch 110/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 250ms/step - accuracy: 0.9519 - loss: 0.1218 - val_accuracy: 0.9421 - val_loss: 0.1587 - learning_rate: 1.0000e-04
    Epoch 111/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 249ms/step - accuracy: 0.9573 - loss: 0.1242 - val_accuracy: 0.9432 - val_loss: 0.1586 - learning_rate: 1.0000e-04
    Epoch 112/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 246ms/step - accuracy: 0.9521 - loss: 0.1141 - val_accuracy: 0.9432 - val_loss: 0.1585 - learning_rate: 1.0000e-04
    Epoch 113/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 246ms/step - accuracy: 0.9518 - loss: 0.1186 - val_accuracy: 0.9378 - val_loss: 0.1669 - learning_rate: 1.0000e-04
    Epoch 114/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 249ms/step - accuracy: 0.9516 - loss: 0.1244 - val_accuracy: 0.9421 - val_loss: 0.1607 - learning_rate: 1.0000e-04
    Epoch 115/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 248ms/step - accuracy: 0.9554 - loss: 0.1124 - val_accuracy: 0.9453 - val_loss: 0.1595 - learning_rate: 1.0000e-04
    Epoch 116/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 249ms/step - accuracy: 0.9466 - loss: 0.1312 - val_accuracy: 0.9378 - val_loss: 0.1639 - learning_rate: 1.0000e-04
    Epoch 117/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 248ms/step - accuracy: 0.9521 - loss: 0.1189 - val_accuracy: 0.9421 - val_loss: 0.1621 - learning_rate: 1.0000e-04
    Epoch 118/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 248ms/step - accuracy: 0.9528 - loss: 0.1155 - val_accuracy: 0.9421 - val_loss: 0.1604 - learning_rate: 1.0000e-04
    Epoch 119/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 247ms/step - accuracy: 0.9549 - loss: 0.1122 - val_accuracy: 0.9400 - val_loss: 0.1599 - learning_rate: 1.0000e-04
    Epoch 120/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m58s[0m 254ms/step - accuracy: 0.9499 - loss: 0.1227 - val_accuracy: 0.9432 - val_loss: 0.1589 - learning_rate: 1.0000e-04
    Epoch 121/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 253ms/step - accuracy: 0.9484 - loss: 0.1311 - val_accuracy: 0.9378 - val_loss: 0.1618 - learning_rate: 1.0000e-04
    Epoch 122/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 253ms/step - accuracy: 0.9577 - loss: 0.1112 - val_accuracy: 0.9400 - val_loss: 0.1632 - learning_rate: 1.0000e-04
    Epoch 123/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 248ms/step - accuracy: 0.9546 - loss: 0.1122 - val_accuracy: 0.9443 - val_loss: 0.1570 - learning_rate: 1.0000e-04
    Epoch 124/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 248ms/step - accuracy: 0.9570 - loss: 0.1175 - val_accuracy: 0.9443 - val_loss: 0.1578 - learning_rate: 1.0000e-04
    Epoch 125/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 252ms/step - accuracy: 0.9500 - loss: 0.1314 - val_accuracy: 0.9421 - val_loss: 0.1599 - learning_rate: 1.0000e-04
    Epoch 126/300
    [1m227/227[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 252ms/step - accuracy: 0.9559 - loss: 0.1166 - val_accuracy: 0.9400 - val_loss: 0.1636 - learning_rate: 1.0000e-04
    Epoch 126: early stopping
    Restoring model weights from the end of the best epoch: 96.



```python
# Create dataframe of model fit history.
mod_dcnn_hist_df = pd.DataFrame().from_dict(mod_dcnn_hist.history, orient = 'columns')

plot_TF_training_history(mod_dcnn_hist_df, 'DCNN - Training History')
```


    
![png](Periocular_Sex_Classification_files/Periocular_Sex_Classification_96_0.png)
    


Impressive we can see that this model trains a little quicker than the previous one, ending at epoch 96. It also seems to just slightly overfit the training data but does utilize batch normalization between each convolution layer so the effects are mitigated.

Similarly we see a lot of oscillation in the performance in the beginning, eventually stabilizing around epoch 80.


```python
y_pred_val_dcnn_proba = mod_dcnn.predict(np.stack(X_val.norm_flat.apply(flat_to_array)), verbose = "auto", callbacks = None)
y_pred_val_dcnn = y_pred_val_dcnn_proba.argmax(axis = 1)

accuracy_score(y_val, y_pred_val_dcnn)
```

    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 92ms/step





    0.9453376205787781



94.53 accuracy on the validation set, just barely beating out the previous model. It will be interesting to see how they compare on the test set.


```python
y_pred_dcnn_proba = mod_dcnn.predict(np.stack(X_test.norm_flat.apply(flat_to_array)), verbose = "auto", callbacks = None)
y_pred_dcnn = y_pred_dcnn_proba.argmax(axis = 1)
```

    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 81ms/step


##### Non-Augmented Training

Train the same model but use the training set that is not augmented/transformed for comparison at the end. Muted outputs for clarity.

*Note: For a fair comparison, the learning rate of the optimizer was lowered from the default: 0.01 to 0.005 as the model was quickly overfitting the training data and the model was very unstable.*


```python
mod_dcnn_no_aug = build_dcnn(input_shape=(img_size, img_size, 3))

optimizer_param = tf.keras.optimizers.SGD(learning_rate = 0.005)
mod_dcnn_no_aug.compile(loss = loss_fx,
                optimizer = optimizer_param,
                metrics = ['accuracy'])

# Callbacks to use (same as above):
early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 30, verbose = 1, restore_best_weights = True)
reduce_lr_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 5, cooldown = 8, min_lr = 0.0001, verbose = 1)

mod_dcnn_no_aug_hist = mod_dcnn_no_aug.fit(ds_train,
                           batch_size = BATCH_SIZE,
                           epochs = n_epochs,
                           verbose = 0,
                           callbacks = [early_stop, reduce_lr_plateau],
                           validation_split = 0.0,
                           validation_data = ds_val,
                           shuffle = True,
                           class_weight = None,
                           sample_weight = None,
                           initial_epoch = 0,
                           steps_per_epoch = None,
                           validation_steps = None,
                           validation_batch_size = None,
                           validation_freq = val_freq)

y_pred_val_dcnn_no_aug_proba = mod_dcnn_no_aug.predict(ds_val, verbose = 0, callbacks = None)
y_pred_val_dcnn_no_aug = y_pred_val_dcnn_no_aug_proba.argmax(axis = 1)

y_pred_dcnn_no_aug_proba = mod_dcnn_no_aug.predict(np.stack(X_test.norm_flat.apply(flat_to_array)), verbose = 0, callbacks = None)
y_pred_dcnn_no_aug = y_pred_dcnn_no_aug_proba.argmax(axis = 1)
```

    
    Epoch 12: ReduceLROnPlateau reducing learning rate to 0.0024999999441206455.
    
    Epoch 24: ReduceLROnPlateau reducing learning rate to 0.0012499999720603228.
    
    Epoch 36: ReduceLROnPlateau reducing learning rate to 0.0006249999860301614.


###### [Back to Table of Contents](#toc)

## 8. Model Analysis and Feature Extraction Discussion: <a name="analysis"></a>

Now that the models have all been successfully trained and tuned, we can extract some fun and interesting information from different sections of the networks.

*Note: Unless otherwise specified all the analysis will be from the model (named CNN in this project) from section 7.2.2.* 

### 8.1. Visualize Model Filters: <a name="filters"></a>

By extracting the weights from the convolution layers, we can visualize the filter being applied to the input images. Here, this process is done to the first convolution layer, visualizing all 16 of the 3x3 kernels. For convenience, I plotted both the separated the RGB channels and the combined RGB filter.


```python
# Extract weights from first convolution layer. 
filters, biases = mod_cnn.layers[0].get_weights()
# Normalize values.
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
print('Filter Shape:', filters.shape)

# Plot all filters and channels.
n_filters = filters.shape[3]
fig, ax = plt.subplots(4, n_filters, sharey = True, sharex = True)

for i in range(n_filters):
    filters_rgb = filters[:,:,:,i]
    for j in range(3):
        ax[j,i].imshow(filters_rgb[:,:,j], cmap = 'gray')
        ax[j,i].set_xticks([])
        ax[j,i].set_yticks([])
        if i == 0:
            ax[j,i].set_ylabel(j)
    ax[3,i].imshow(filters_rgb)
    ax[3,i].set_xticks([])
    ax[3,i].set_yticks([])
    ax[3,i].set_xlabel(i)

ax[3,0].set_ylabel('RGB')
fig.supylabel('Channel Index')
fig.supxlabel('Filter Index', y = 0.25)
fig.suptitle('First Convolution Layer Filters', y = 0.7)
fig.tight_layout(h_pad = -14, w_pad = 0.5)
plt.show()
```

    Filter Shape: (3, 3, 3, 16)



    
![png](Periocular_Sex_Classification_files/Periocular_Sex_Classification_106_1.png)
    


Note, pixels that are weighted more show up as white and pixels that are not show up as black.

Very similar to sobel filters and the like, these filters have been trained and weighted to pick up simple features like horizontal, vertical lines, angled lines, etc but of course these are represented in RGB space so it is a little more difficult to visually identify their purpose.


For a more human-comprehendible visualization -- next, we can jump ahead in the network to visualize the output of the convolution blocks as a whole by accessing the output at the max pool layer that is placed at the end of each convolution block. By passing through an image into the network, instead of the individual kernels we just saw, these outputs equate to the down-sampled filter view of the image.

The code will output as follows:
1. Take an unseen image (outside of all train/val/test datasets) and pass it through the model, recording the output at each layer.
    - Plot the unseen image for reference.
2. The second plot will be from the max pooling layer (layer 4) at the end of first block of 2 stacked 3x3 convolution layers.
    - This should output 16 - 28 x 28 weighted filters.
3. The third plot is from the max pooling layer (layer 11) at the end of the 2nd block of 3 stacked 3x3 convolutions.
    - This should output 32 - 14 x 14 weighted filters.


```python
# Indexes of maxpool layers at end of conv blocks.
pool_idx = [4, 11]

# Retrieve output of each layer.
outputs = [mod_cnn.layers[i].output for i in pool_idx]
mod_cnn_feat = tf.keras.Model(inputs = mod_cnn.inputs,
                       outputs = outputs)
```


```python
# Load unseen image.
img = Image.open('./Data/Female/186.jpg')
img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
img = np.asarray(img)
img = np.expand_dims(img, axis = 0)
plt.imshow(img[0,:,:,:])
plt.show()

# Retrieve feature maps for image.
feature_maps = mod_cnn_feat.predict(img)
# Get number of filters from shape of filters = (16, 32).
n_filters = [feature_maps[0].shape[3], feature_maps[1].shape[3]]
# Iterate through each feature map.
for i, fmap in enumerate(feature_maps):
    print(f'Convolution Block {i+1}')
    print('Filter Shape:', fmap.shape)
    fig, ax = plt.subplots(int(n_filters[i]/4), 4, figsize = ((4,4) if i == 0 else (4,8)))
    # Iterate through number of filters in current fmap layer.
    for j in range(n_filters[i]):
        # Plot each filter feature map.
        ax[j//4, j%4].set_xticks([])
        ax[j//4, j%4].set_yticks([])
        ax[j//4, j%4].imshow(fmap[0,:,:,j], cmap = 'gray')

    plt.show()
```


    
![png](Periocular_Sex_Classification_files/Periocular_Sex_Classification_109_0.png)
    


    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step
    Convolution Block 1
    Filter Shape: (1, 28, 28, 16)



    
![png](Periocular_Sex_Classification_files/Periocular_Sex_Classification_109_2.png)
    


    Convolution Block 2
    Filter Shape: (1, 14, 14, 32)



    
![png](Periocular_Sex_Classification_files/Periocular_Sex_Classification_109_4.png)
    


This is a great way to begin understanding what features the network has learned to weight heavily in its classification task.

- In the first convolution block output we can see several filters that capture various structures:
    - The iris and retina.
    - Masking the iris and retina but weight the skin around the eye.
    - Heavily weighting eye and eyelash edges.

- The last plot showing the last convolution block shows what would be expected of an activation map view. We can see a much more contrasted and simplified view of this image. Some filters seem to be capture:
    - Skin creases.
    - Eyebrows.
    - The skin above the eye but below the eyebrow.
    - Eye lashes.
    - General edge detection.
    - Combinations of hair and eyes.

###### [Back to Table of Contents](#toc)

### 8.2. Occlusion Sensitivity Plots: <a name="occlusion"></a>

Another method to determine by how much a model weights certain features is Occlusion Sensitivity. Occlusion sensitivity, put simply, occludes or covers portions of an input image and passes it through the model, tracking the changes in class probabilities and thus building a heatmap of important areas of the image. 

Two parameters are important in this process: The occluding patch size and the process by which the occlusion location is chosen.

You can implement this manually by randomly choosing occlusion locations, eventually processing each position, or even just iterate across the axis. 

In this case, TensorFlow has an addon called `tf-explain` that makes the process easier and is implemented in a better way than the two simpler methods above.

The plots below occlusion sensitivity being performed on 4 unseen images and with 5 different occlusion patch sizes.
- 2x2, 4x4, 8x8, 16x16, 24x24


```python
# List of unseen images to use. 2 Female, 2 Male.
unseen_eyes = ['./Data/Female/186.jpg',
               './Data/Female/2060.jpg',
               './Data/Male/186.jpg',
               './Data/Male/3154.jpg']

fig, ax = plt.subplots(6, 4, figsize = (9, 12))
# Iterate through the 4 random unseen eye images.
for i, file in enumerate(unseen_eyes):
    # From TF-Explain: https://github.com/sicara/tf-explain/blob/master/examples/core/occlusion_sensitivity.py
    img = tf.keras.preprocessing.image.load_img(file, target_size = (56, 56))
    img = tf.keras.preprocessing.image.img_to_array(img)

    explainer = OcclusionSensitivity()
    # Set index for correct class.
    class_index = 0 if i < 2 else 1

    # Iterate different patch sizes and plot.
    for j, patch in enumerate([2,4,8,16,24]):
        # Compute Occlusion Sensitivity at different patch sizes.
        explained = explainer.explain(([img], None), mod_cnn, class_index = class_index, patch_size = patch)
        ax[j, i].set_title(f'{patch}')
        ax[j, i].set_xticks([])
        ax[j, i].set_yticks([])
        ax[j, i].imshow(explained)

    # Show original image at last row.
    img = Image.open(file)
    img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
    ax[5, i].imshow(img)
    ax[5, i].set_xticks([])
    ax[5, i].set_yticks([])
    ax[5, i].set_title(f'Class: {class_index}')
    
plt.show()
```

    [1m25/25[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 17ms/step
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 15ms/step
    [1m2/2[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 14ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 19ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 18ms/step
    [1m25/25[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 18ms/step
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 15ms/step
    [1m2/2[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 12ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 18ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 18ms/step
    [1m25/25[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 17ms/step
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 14ms/step
    [1m2/2[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 13ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 20ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 18ms/step
    [1m25/25[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 17ms/step
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 16ms/step
    [1m2/2[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 12ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 18ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 18ms/step



    
![png](Periocular_Sex_Classification_files/Periocular_Sex_Classification_112_1.png)
    


*Female = 0, Male = 1*

The two left columns are labeled as women and the two right columns are men.

Interestingly, and at least for this small sample, we can see some subtle difference in what features are being shown as important between the two classes. We can also see some changes in results as the occlusion patch size increase, thus showing results of the effects of larger features.

Female:
- Eyebrows are being highlighted more than in the male photos.
- Specifically the upper lid eyelashes are highlighted more here. Especially at the far corner of the eye.
- It may also weight the skin around the eye a little more for this class.

Male:
- A little upper eyelash highlight here as well.
- We also see a high amount of activation along the lower edge of the eye.
- Possibly more highlighting around the iris compared to the women.

This plot brings up a lot of questions and insights about how this model is classifying images. It is quite possible that there are subtle differences in the dataset that are effecting it's "decisions". For instance, it might be useful to explore how balanced the hair color and makeup use is between the two classes. It could be possible that mascara use in the women is allowing the models to heavily favor using this feature for classifying the images. Futhermore, since the source of these images are from a IMDB-esque website most of these images are likely either professional headshots or movie posters. These facts are likely introducing a large amount of bias into this dataset.

###### [Back to Table of Contents](#toc)

### 8.3. Principal Component Analysis (PCA) of Feature Embeddings: <a name="pca"></a>

Another valuable analysis tool is to extract the feature embeddings (layer outputs) and use Principal Component Analysis (PCA) to reduce the dimensions and plot the top two principal components. We can even plot each image on this PCA for an impressive visualization that might allow us to identify emerging complex patterns.

The images from the test set will be used for these plots and each image will be bordered with the color of the true class label (Female = Teal, Male = Purple). One version with the image and another as a scatter plot for easier visual parsing of the class distribution.


```python
# Get feature vector from model.
def feature_extractor(model, layer_num):
    # https://keras.io/getting_started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer-feature-extraction
    extractor = tf.keras.Model(inputs = model.inputs,
                            outputs = [layer.output for layer in model.layers])
    features_test = extractor(np.stack(X_test.norm_flat.apply(flat_to_array)))

    feature_vec_cnn = []
    # Take feature vector from layer_num.
    for i in range(len(X_test)):
        feature_vec_cnn.append(features_test[layer_num][i].numpy())
    
    return feature_vec_cnn
```


```python
def pca_plot(features, image_arrays, zoom = 0.5, cmap = None, photos = True): #zoom = 0.125

    # Reduce dimensions to 2 using PCA.
    pca = PCA(n_components = 2)
    pca_fit = pca.fit_transform(features)
    print(f'Explained Variance Ratios: {pca.explained_variance_ratio_}')

    _, ax = plt.subplots(figsize = (20, 15), subplot_kw = {'aspect' : 'equal'})
    ax.scatter(pca_fit[:, 0], pca_fit[:, 1], c = cmap, alpha = 0.8)
    
    # Add eye photos to plot.
    if photos == True:
        for i, rgb_flat in enumerate(image_arrays):
            # Load image.
            image = Image.fromarray(rgb_flat.reshape(img_size,img_size,3))
            # Zoom out.
            im = OffsetImage(image, zoom = zoom)
            # Set class label color for edge bbox.
            bboxprops = ({'edgecolor' : cmap[i], 'lw' : 2} if cmap is not None else None)
            anno_bbox = AnnotationBbox(offsetbox = im, 
                                xy = pca_fit[i],
                                xycoords = 'data',
                                frameon = (bboxprops is not None),
                                pad = 0.075,
                                bboxprops = bboxprops)
            ax.add_artist(anno_bbox)

    ax.set_axis_off()
    ax.axis('tight')
    ax.set_title('Principal Component Analysis (PCA) of Eye Feature Embeddings')

    return ax
```


```python
# Teal = Female, Purple = Male
cmap_sex_all = ['#439A86' if sex == 0 else '#423e80' for sex in y_test]
```

First from the CNN (section 7.2.2.) we can extract and visualize the resulting embeddings coming out of the last convolution block and into the first dense layer.

Above each plot is the layer number and which model's embedding is being shown. Also, the resulting top 2 principal component's explained variance ratios.


```python
feature_vec_cnn = feature_extractor(mod_cnn, 13)

# Plot 2nd to last fully connected layer.
print('CNN - Layer 13')
_ = pca_plot(feature_vec_cnn, X_test.rgb_flat.tolist(), cmap = cmap_sex_all)
plt.show()

_ = pca_plot(feature_vec_cnn, X_test.rgb_flat.tolist(), cmap = cmap_sex_all, photos = False)
plt.show()
```

    CNN - Layer 13
    Explained Variance Ratios: [0.20404717 0.09216744]



    
![png](Periocular_Sex_Classification_files/Periocular_Sex_Classification_119_1.png)
    


    Explained Variance Ratios: [0.20404717 0.09216744]



    
![png](Periocular_Sex_Classification_files/Periocular_Sex_Classification_119_3.png)
    


Identifying combine characteristics from principal components is a highly imprecise art, but I believe we can see a few things along each axis here.

- Along the x-axis is the class label, of course, with two easily identifiable clusters meeting and slightly overlapping at the center.
- The y-axis is a little more subtle, but here it seems the lower on the y-axis shows eyes with heavier dark features (eye shadow or mascara for the women or darker lighting or hair color for the men).

Now, we can do the same but on the last fully connected layer (dense) of the same model.


```python
feature_vec_cnn = feature_extractor(mod_cnn, 16)

# Plot the last fully connected layer PCA.
print('CNN - Layer 16')
_ = pca_plot(feature_vec_cnn, X_test.rgb_flat.tolist(), cmap = cmap_sex_all)
plt.show()

_ = pca_plot(feature_vec_cnn, X_test.rgb_flat.tolist(), cmap = cmap_sex_all, photos = False)
plt.show()
```

    CNN - Layer 16
    Explained Variance Ratios: [0.54373284 0.03505089]



    
![png](Periocular_Sex_Classification_files/Periocular_Sex_Classification_121_1.png)
    


    Explained Variance Ratios: [0.54373284 0.03505089]



    
![png](Periocular_Sex_Classification_files/Periocular_Sex_Classification_121_3.png)
    


The difference between this and the previous is drastic and the two clusters seem to have put more distance between them. Oddly we can see a new prevailing feature along the y-axis here.

- On the y-axis, we now see that low y values show images of individual's left eye while higher on the y-axis are right eyes. Oddly I don't see this to be the case for the Males.
- The men have a much more dense cluster showing little variance in distance along both axis.


Now to do the same thing but with the DCNN (section 7.2.3.). We will be extracting the output from the `GlobalAveragePooling2D` layer near the bottom of the network before prediction.


```python
feature_vec_dcnn = feature_extractor(mod_dcnn, -2)

print('DCNN - Layer 123')
# Plot global pool layer.
_ = pca_plot(feature_vec_dcnn, X_test.rgb_flat.tolist(), cmap = cmap_sex_all)
plt.show()

_ = pca_plot(feature_vec_dcnn, X_test.rgb_flat.tolist(), cmap = cmap_sex_all, photos = False)
plt.show()
```

    DCNN - Layer 123
    Explained Variance Ratios: [0.77782455 0.10590087]



    
![png](Periocular_Sex_Classification_files/Periocular_Sex_Classification_123_1.png)
    


    Explained Variance Ratios: [0.77782455 0.10590087]



    
![png](Periocular_Sex_Classification_files/Periocular_Sex_Classification_123_3.png)
    


The explained variance ratios a show an impressive ~87% between the two components and its easy to see that by how structured this PCA plot turned out. The easiest learners are at the edge of the long swooping shape.

###### [Back to Table of Contents](#toc)

### 8.4. Misclassification Exploration: <a name="misclass"></a>

It can be useful to examine the images that were misclassified and see if there are any obvious image differences or correlations between them that might give insights into why the model is misclassifying them.


```python
# Get mask for all misclassified images in test set.
mask_missed_cnn = y_test != y_pred_cnn
# Sample 25 images from misclassified images.
missed_real_labels = y_test[mask_missed_cnn].sample(25, random_state = 11)

cmap_sex_missed = ['#439A86' if sex == 0 else '#423e80' for sex in missed_real_labels]
class_map = {0: 'Female', 1: 'Male'}

fig, ax = plt.subplots(5, 5, figsize = (10,10))
# Use index to retrieve images and loop through.
for i, img in enumerate(X_test.loc[missed_real_labels.index].rgb_flat):
    ax[i//5, i%5].set_xticks([])
    ax[i//5, i%5].set_yticks([])
    ax[i//5, i%5].imshow(flat_to_array(img))
    # Add border for real label.
    border = plt.Rectangle((0, 0), 55, 55,
                           fill = False,
                           color = cmap_sex_missed[i],
                           linewidth = 3.5)
    ax[i//5, i%5].add_patch(border)
    # Add text for incorrect predictions.
    ax[i//5, i%5].text(2, 53, 'Prediction:', fontsize = 9, color = '#e61405')
    ax[i//5, i%5].text(53, 53, (f"{class_map[1 - missed_real_labels.iloc[i]]}"),
                       fontsize = 9, 
                       weight = 'bold',
                       ha = 'right', 
                       color = cmap_sex[class_map[1 - missed_real_labels.iloc[i]]]
                       ).set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'),
                                           path_effects.Normal()])
plt.show()
```


    
![png](Periocular_Sex_Classification_files/Periocular_Sex_Classification_127_0.png)
    


**Real Label:** The colored border of the image indicates the real class, consistent with all other visualizations.

- Teal = Female
- Purple = Male

**Predicted Label:** The text printed on the image.

Looking through these sampled 25 (out of 29) of misclassified images, many of them share a few common characteristics.

- Hair
- Age (wrinkles)
- Weird angles
- Glasses
- Subtle eyelashes
- No eyebrows

Besides the obvious issues like the images with hair obscuring the eyes and images taken at odd angles, there likely aren't enough images in the training data that represent older individuals or people with glasses. There also seem to be quite a few examples of people with light colored or hard to see eyelashes. These hard learners (and others) might benefit from higher resolution images.

###### [Back to Table of Contents](#toc)

### 8.5. Predict for Fun: <a name="fun"></a>

Now, it seems like a missed opportunity to not attempt to predict on some personal photos. 

Below I will import a periocular image of myself and my wife. We both present as Male and Female respectively. Let's see if my model can correctly classify us as such.


```python
# Import prediction images for fun.
predict_files = os.listdir('./Data/Predict/')

fig, ax = plt.subplots(1, len(predict_files))
for i, img in enumerate(predict_files):
    img = Image.open(f'./Data/Predict/{img}')
    img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
    img = img.convert('RGB')
    img = np.asarray(img)
    img = np.expand_dims(img, axis = 0)
    ax[i].imshow(img[0,:,:,:])
    # Predict image class.
    predict_proba = mod_cnn.predict(img, verbose = "auto", callbacks = None)
    predict = predict_proba.argmax(axis = 1)
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    # Add border for real label.
    border = plt.Rectangle((0, 0), 55, 55, fill = False, color = cmap_sex[class_map[predict[0]]], linewidth = 3.5)
    ax[i].add_patch(border)
    ax[i].text(2, 53, 'Prediction:', fontsize = 10, color = '#e61405')
    ax[i].text(54, 53, (f"{class_map[predict[0]]}"), fontsize = 10, weight = 'bold', ha = 'right', 
                       color = cmap_sex[class_map[predict[0]]]
                       ).set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'),
                                           path_effects.Normal()])
plt.show()
```

    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 14ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 13ms/step



    
![png](Periocular_Sex_Classification_files/Periocular_Sex_Classification_130_1.png)
    


Well, no gender-bending surprises there, the model works great!

###### [Back to Table of Contents](#toc)

## 9. Results <a name="results"></a>

---

Finally, we can check the performance of each model on the test set.


```python
# Plot confusion matrices of all models.
def confusion_matrix_subplot(y_pred_dict):
    model_class_vals = {}

    fig, ax = plt.subplots(int(np.ceil(len(y_pred_dict) / 2)), 2, figsize = (7, 7), sharey = True)
    for i, (model, y_pred) in enumerate(y_pred_dict.items()):
        y_true = y_test

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix = cm)
        disp.plot(ax = ax[i//2, i%2])
        disp.ax_.set_title(str(model))
        disp.im_.colorbar.remove()
        # Store classification metrics
        # In the order [tn, fp, fn, tp]
        model_class_vals[model] = list(confusion_matrix(y_true, y_pred, labels = y_true.unique().sort()).ravel())

    # Check if odd number of models and delete last subplot if true.
    if len(y_pred_dict) % 2 != 0:
        fig.delaxes(ax[i//2, (len(y_pred_dict) % 3)])
    fig.tight_layout()
    plt.subplots_adjust(wspace = 0.2, hspace = 0)
    fig.colorbar(disp.im_, ax = ax)
    fig.suptitle('Confusion Matrices of All Models')
    
    return model_class_vals

# Plot all model ROC curves.
def roc_curve_subplot(y_pred_proba_dict):
    fig, ax = plt.subplots(1,1, figsize = (8,8))
    for i, model in enumerate(y_pred_proba_dict.keys()):
        y_pred_proba_temp = y_pred_proba_dict[model][:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba_temp)
        auc_score = auc(fpr, tpr)
        ax.plot(fpr, tpr, label = f"AUC = {auc_score:0.3f} | {model}", alpha = 0.8)

    # Add 45 degree line for 0.5 AUC.
    ax.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), 'k--')

    ax.set_title('Test Set - ROC Curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    plt.grid(alpha = 0.4)
    plt.legend(loc = 'center right')

    plt.show()

# Highlight the best model's test results green at each proportion.
def max_value_highlight(df):
    max_test_rows = df.max()
    is_max = (df == max_test_rows)
    
    return ['background-color:green' if v else '' for v in is_max]

# Highlight the top two results in each column blue so that 2nd place is in blue after .apply().
def highlight_top_two(df):
    # Sort values
    sorted_df = df.sort_values(ascending = False)
    top_two = sorted_df.iloc[:2]
    # Mask
    is_top_two = df.isin(top_two)

    return ['background-color: blue' if v else '' for v in is_top_two]
```

For convenience, we can store the prediction arrays in dictionaries to be iterated over within the functions above.


```python
y_pred_dict = {'KNN': y_pred_knn,
               'FNN': y_pred_fnn,
               'CNN': y_pred_cnn,
               'DCNN': y_pred_dcnn}

y_pred_proba_dict = {'KNN': y_pred_knn_proba,
                     'FNN': y_pred_fnn_proba,
                     'CNN': y_pred_cnn_proba,
                     'DCNN': y_pred_dcnn_proba}

y_pred_val_dict = {'KNN': y_pred_val_knn,
               'FNN': y_pred_val_fnn,
               'CNN': y_pred_val_cnn,
               'DCNN': y_pred_val_dcnn}

y_pred_no_aug_dict = {'KNN': y_pred_knn,
               'FNN': y_pred_fnn_no_aug,
               'CNN': y_pred_cnn_no_aug,
               'DCNN': y_pred_dcnn_no_aug}

y_pred_val_no_aug_dict = {'KNN': y_pred_val_knn,
               'FNN': y_pred_val_fnn_no_aug,
               'CNN': y_pred_val_cnn_no_aug,
               'DCNN': y_pred_val_dcnn_no_aug}
```

Here I can plot a confusion matrix for each model.


```python
model_class_vals = confusion_matrix_subplot(y_pred_dict)
```


    
![png](Periocular_Sex_Classification_files/Periocular_Sex_Classification_137_0.png)
    


The performance of the CNN and DCNN are almost the same, though the CNN edges out the DCNN here just barely on the test set (which was the opposite on the validation set). It is also surprising to see that the two CNN-based models perform about equally on each class.

KNN:
- 155 misclassifications, more Female than Male.

FNN:
- 95 misclassifications, more Male than Female.

CNN:
- 29 misclassifications, about equal.

DCNN:
- 31 misclassifications, about equal.


```python
roc_curve_subplot(y_pred_proba_dict)
```


    
![png](Periocular_Sex_Classification_files/Periocular_Sex_Classification_139_0.png)
    


The Area Under the Curve (AUC) scores of each of the models shows the same story here -- though the DCNN just edged out the CNN, but in reality the both performed about the same on the test set.


Now let's calculate the accuracy of each model and output the results into a dataframe. Also, we will evaluate each model on the non-augmented training dataset.

Accuracy was chosen because the dataset was class balanced and that was maintained through splitting into testing and validation sets as well.


```python
# Non-augmented training images.
results_no_aug_dict = {}
for model, y_pred in y_pred_no_aug_dict.items():
        val_acc = accuracy_score(y_val, y_pred_val_no_aug_dict[model])
        test_acc = accuracy_score(y_test, y_pred)
        results_no_aug_dict[model] = [val_acc, test_acc]

results_no_aug_df = pd.DataFrame().from_dict(results_no_aug_dict, orient = 'index', columns = ['Val Accuracy', 'Test Accuracy'])
results_no_aug_df = pd.concat([pd.DataFrame({'Val Accuracy': 0.5, 'Test Accuracy': 0.5}, index = ['Random Baseline']), results_no_aug_df])
results_no_aug_df.loc['KNN'] = [0, 0]
#results_no_aug_df.style.apply(highlight_top_two).apply(max_value_highlight)

# Augmented training images.
results_dict = {}
for model, y_pred in y_pred_dict.items():
        val_acc = accuracy_score(y_val, y_pred_val_dict[model])
        test_acc = accuracy_score(y_test, y_pred)
        results_dict[model] = [val_acc, test_acc]

results_df = pd.DataFrame().from_dict(results_dict, orient = 'index', columns = ['Val Accuracy', 'Test Accuracy'])
results_df = pd.concat([pd.DataFrame({'Val Accuracy': 0.5, 'Test Accuracy': 0.5}, index = ['Random Baseline']), results_df])
#results_df.style.apply(highlight_top_two).apply(max_value_highlight)
```


```python
results_concat = pd.concat([results_df, results_no_aug_df], axis = 1)
results_concat.columns = pd.MultiIndex.from_tuples(zip(['Augmented', 'Augmented.', 'Not Augmented', 'Not Augmented.'], results_concat.columns))
results_concat.style.apply(highlight_top_two).apply(max_value_highlight)
```




<style type="text/css">
#T_09a64_row3_col0, #T_09a64_row4_col1, #T_09a64_row4_col2, #T_09a64_row4_col3 {
  background-color: blue;
}
#T_09a64_row3_col1, #T_09a64_row3_col2, #T_09a64_row3_col3, #T_09a64_row4_col0 {
  background-color: blue;
  background-color: green;
}
</style>
<table id="T_09a64">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_09a64_level0_col0" class="col_heading level0 col0" >Augmented</th>
      <th id="T_09a64_level0_col1" class="col_heading level0 col1" >Augmented.</th>
      <th id="T_09a64_level0_col2" class="col_heading level0 col2" >Not Augmented</th>
      <th id="T_09a64_level0_col3" class="col_heading level0 col3" >Not Augmented.</th>
    </tr>
    <tr>
      <th class="blank level1" >&nbsp;</th>
      <th id="T_09a64_level1_col0" class="col_heading level1 col0" >Val Accuracy</th>
      <th id="T_09a64_level1_col1" class="col_heading level1 col1" >Test Accuracy</th>
      <th id="T_09a64_level1_col2" class="col_heading level1 col2" >Val Accuracy</th>
      <th id="T_09a64_level1_col3" class="col_heading level1 col3" >Test Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_09a64_level0_row0" class="row_heading level0 row0" >Random Baseline</th>
      <td id="T_09a64_row0_col0" class="data row0 col0" >0.500000</td>
      <td id="T_09a64_row0_col1" class="data row0 col1" >0.500000</td>
      <td id="T_09a64_row0_col2" class="data row0 col2" >0.500000</td>
      <td id="T_09a64_row0_col3" class="data row0 col3" >0.500000</td>
    </tr>
    <tr>
      <th id="T_09a64_level0_row1" class="row_heading level0 row1" >KNN</th>
      <td id="T_09a64_row1_col0" class="data row1 col0" >0.773848</td>
      <td id="T_09a64_row1_col1" class="data row1 col1" >0.750804</td>
      <td id="T_09a64_row1_col2" class="data row1 col2" >0.000000</td>
      <td id="T_09a64_row1_col3" class="data row1 col3" >0.000000</td>
    </tr>
    <tr>
      <th id="T_09a64_level0_row2" class="row_heading level0 row2" >FNN</th>
      <td id="T_09a64_row2_col0" class="data row2 col0" >0.843516</td>
      <td id="T_09a64_row2_col1" class="data row2 col1" >0.847267</td>
      <td id="T_09a64_row2_col2" class="data row2 col2" >0.863880</td>
      <td id="T_09a64_row2_col3" class="data row2 col3" >0.861736</td>
    </tr>
    <tr>
      <th id="T_09a64_level0_row3" class="row_heading level0 row3" >CNN</th>
      <td id="T_09a64_row3_col0" class="data row3 col0" >0.944266</td>
      <td id="T_09a64_row3_col1" class="data row3 col1" >0.953376</td>
      <td id="T_09a64_row3_col2" class="data row3 col2" >0.931404</td>
      <td id="T_09a64_row3_col3" class="data row3 col3" >0.914791</td>
    </tr>
    <tr>
      <th id="T_09a64_level0_row4" class="row_heading level0 row4" >DCNN</th>
      <td id="T_09a64_row4_col0" class="data row4 col0" >0.945338</td>
      <td id="T_09a64_row4_col1" class="data row4 col1" >0.950161</td>
      <td id="T_09a64_row4_col2" class="data row4 col2" >0.908896</td>
      <td id="T_09a64_row4_col3" class="data row4 col3" >0.885852</td>
    </tr>
  </tbody>
</table>




In each column, the best metric is highlighted in green and second best is blue. The two left columns are the models trained with the image augmentations and the two right columns are without augmentation.

First looking at the models trained using image augmentations:
- Again, the CNN and DCNN are performing almost exactly the same, switching between the validation and test sets.
- The CNN outperforms the DCNN on the test set by ~0.3% with a test accuracy of ~95.34%.
- The DCNN performed 2nd best with an accuracy score of ~95.02%.
- Each model outperformed the KNN baseline and random chance metrics.

Non-Augmented:
- The FNN performs better on the non-augmented training images. This makes sense because the FNN, using only dense layers, is not invariant to image transformations and the images from the unaltered dataset are roughly located in the same location/orientation. Thus, the FNN struggles to generalize between a more diverse dataset in a computer vision context.
- The CNN performs the best of all the models with this unaltered data, but worse here than when using the augmented images.
- The DCNN accuracy scores fell 4-7% between the augmented and non-augmented datasets. This much deeper model benefits from the much (effectively) larger and diverse dataset that the augmented training images provides.

###### [Back to Table of Contents](#toc)

## 10. Conclusion <a name="conclusion"></a>

---

The results and analysis from this project demonstrate the effectiveness of deep convolutional neural networks (DCNNs) for classifying sex based on periocular images. An accuracy score of approximately 95% for sex identification using low-resolution 56x56 images shows promising results. The custom-built CNN performed the best on the test set with an accuracy score of ~95.34, while the DCNN modeled after ResNet-50 closely followed and outperformed on the validation set.

**Some key takeaways from this project:**

**DCNN Efficacy:** By training models of various architectures, it confirms that DCNNs can extract essential features from the periocular region to differentiate between sexes, even with shallower networks. The deep models displayed robust performance, especially after fine-tuning, and random image augmentation of the training data.

**Feature Extraction Insights:** In-depth analysis of model filters and feature maps revealed that certain filters in the early convolutional layers focused on basic visual features such as edges and textures, while deeper layers captured more complex patterns and structures relating to the eye and periocular region. The hierarchical layering of the network's architectures yield effective feature extraction and accurate sex classification. By visualizing convolutional layer filters, it was observed that the model learned distinct features which contributed to classification performance. These insights into class-specific features that the models leveraged for classification, hint that the techniques and images used within this project may have models that are predicting gender instead of sex -- further work is necessary to fully determine this.

**Data Augmentation's Impact:** CNN-based models trained with image augmentation outperformed those without augmentation, showcasing improved generalization and reduced overfitting by adding size and diversity to the dataset. This was particularly evident because the dataset's size is relatively small for deep learning applications.

Not only were the models effective at sex classification using periocular images, but valuable insights were gained by exploring the feature extractions, revealing how networks process periocular images. The comparison of models with and without data augmentation also highlight the importance of using a sufficiently large and diverse dataset in deep learning. This project represents a strong first step in sex classification using eye images and informs the next steps for further understanding this task.

### 10.1. Limitations: <a name="limitations"></a>

- The analysis from section 8 raises important questions about how the model is classifying eye images, suggesting that subtle dataset differences may be influencing its decisions. For example, it may be worth investigating whether hair color and makeup such as mascara in women's images could be a dominant feature for classification. 
- The dataset presents a likely heavily biased distribution of not only eye presentations (makeup, Photoshop touchups, etc), but also a skewed distribution in age and race of the individual.
- Image resolution. Higher resolution images would likely improve performance and allow further analysis of features extracted.

###### [Back to Table of Contents](#toc)

### 10.2. Future Work: <a name="future"></a>

- A larger dataset would help alleviate some of the issues described above and possibly help models classify those hard misclassified examples.
- Find or compile a higher resolution dataset.
- Instead of using images of the periocular region, segment the iris and attempt to classify only using that masked segmentation. This has been shown to work but many current studies show biased methodologies and end with conflicting conclusions on this subject. It seems data leakage, poor segmentation leading to other characteristics to bleed into the models, tends to be the biggest issue with current studies.
- Further explore model interpretability to better understand whether the model is relying heavily on gender characteristics in the periocular region.
    - Perform more tests to confound these models to determine how much weight is given to characteristics like makeup or eyelash contrast vs eye and eyebrow shape.

###### [Back to Table of Contents](#toc)

## Appendix A - Online References: <a name="appendixa"></a>

Resources that helped along the way in no particular order.

1. 2016 paper iris sex classification based on in-depth feature selection and SVM: https://ieeexplore.ieee.org/abstract/document/7447785
2. 2018 paper iris sex classification based on Zernike moments and classifying using SVM and KNN: https://ieeexplore.ieee.org/document/8492757
3. 2019 paper iris .. using CNNs and image augmentation: https://ieeexplore.ieee.org/abstract/document/7447785
4. 2023 paper classifcation using periocular region and iris using pre-trained CNNs and transfer learning: https://www.sciencedirect.com/science/article/pii/S2666307423000268
5. 2019 Paper refuting previous papers claims on classifications based solely on iris; claiming periocular features were included: https://ieeexplore.ieee.org/document/8659186
6. Great examples of utilizing matplotlib to plot PCA/t-SNE tied to their images https://www.kaggle.com/code/hmendonca/proper-clustering-with-facenet-embeddings-eda
7. Local Binary Patterns used for feature extraction: https://en.wikipedia.org/wiki/Local_binary_patterns
8. Sobel filter edge detector: https://en.wikipedia.org/wiki/Sobel_operator

Alternative Datasets that would improve this project and allow for more nuanced segmentation of eye features:
1. Biometrics Research of Notre Dame has a high quality dataset. Behind a license application: https://cvrl.nd.edu/projects/data/
2. Another dataset with high-resolution iris photos. Behind license and too big for this project, though: https://ieee-dataport.org/documents/iris-super-resolution-dataset

Exported to HTML via command line using:

- `jupyter nbconvert Periocular_Sex_Classification.ipynb --to html`
- `jupyter nbconvert Periocular_Sex_Classification.ipynb --to html --HTMLExporter.theme=dark`

###### [Back to Table of Contents](#toc)
