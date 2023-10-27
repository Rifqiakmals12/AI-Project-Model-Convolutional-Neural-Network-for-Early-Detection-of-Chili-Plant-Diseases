# Model Convolutional Neural Network for Early Detection of Chili Plant Diseases in Small Datasets

## Table of Contents
  - [About](#about)
  - [Program Description](#program-description)
  - [Datasets Used](#datasets-used)
  - [Method](#method)
  - [Results](#results)
  - [Conclusion](#conclusion)
  - [Contact](#contact)

## About

At the end of 2021 in Indonesia, there was an increase in the price of basic foodstuffs, including chili. This price increase was due to farmers' chili supply shortage because diseases attacked chili plants in various areas. Early detection of chili plant diseases is essential to maintain the quality and productivity of crop yields. Research related to the detection of chili plant diseases has been developed by many researchers, for example, using machine learning techniques. Previous research used machine learning to classify three classes of chili diseases. The results of this research are not optimal because the amount of data used is small, so it only reaches 86% accuracy. Therefore, we propose using the Convolutional Neural Network (CNN) method, which is part of deep learning.

## Program Description

This program was created to build a CNN architectural model for processing small datasets. We developed the CNN architecture to process small amounts of data. The dataset consists of 5 classes: healthy, leaf curl, leaf spot, whitefly, and yellowish. The raw data obtained is preprocessed before going to the feature extraction stage. The preprocessing process in this research is to homogenize the size and augment the data. The results of the preprocessing data will be used for feature extraction and classification using CNN. 

## Datasets Used

- [Dataset](https://www.kaggle.com/datasets/dhenyd/chili-plant-disease): This model used a dataset sourced from Kaggle. This research uses a dataset in the form of chili diseases in Indonesia. The chili disease classification that we use consists of 5 classes: healthy, leaf curl, leaf spot, whitefly, and yellowish Example of the dataset :
  
![image](https://github.com/Rifqiakmals12/AI-Project-Model-Convolutional-Neural-Network-for-Early-Detection-of-Chili-Plant-Diseases/assets/72428679/b99476b8-5ca4-4cca-96d0-912c44fb55fd)

## Method
The chili plant disease classification begins with inputting chili disease image data along with the identity of the disease, which is used as a label. The input data is divided into two parts: training data and validation data. All data is pre-processed in the form of resizing before being processed to the feature extraction stage using CNN or DenseNet201 models. The result of the feature extraction process using CNN or DenseNet201 is a model file with the extension .h5. The built model is evaluated using the best loss and accuracy values. The best modeling results will be used for evaluation using a confusion matrix. Figure shows a flow diagram forming a chili plant disease classification.  

![image](https://github.com/Rifqiakmals12/AI-Project-Model-Convolutional-Neural-Network-for-Early-Detection-of-Chili-Plant-Diseases/assets/72428679/974deb0a-5e02-44c5-8b00-af8f436286de)


1. __Data Acquisitions :__
  The chili disease classification that we use consists of 5 classes: healthy, leaf curl, leaf spot, whitefly, and yellowish. Each class consists of 116 images. We divide the data into 70% training and 30% training data. 

2. __Data Pre-Processing :__
The dataset being processed to the feature extraction stage using the CNN or transfer learning model is resized to a size of 150 x 150.

3. __System Evaluation :__
The evaluation stage starts by evaluating the results of training validation loss between the two CNN and DenseNet201 models. The lowest validation loss value will be used for testing using the confusion matrix. In addition, we also provide graphs on the training process to determine whether the model made is overfitting or underfitting.

## Results
In this section, we will discuss the results of the study. The discussion starts with the configuration of the dataset, the training results, the confusion matrix test results, and an example of the output of the detection results.

#### A. Dataset Configuration :
The dataset used in this study uses data from Chile Plant Disease. The total image data collected is 580 image data. The data is divided into training and testing data consisting of 400 images as training data 70% and 180 images as data testing 30%. Each training and testing data has five classes: healthy, leaf curl, leaf spot, whitefly, and yellowish. This table shows an example of data sharing used in this research.

![image](https://github.com/Rifqiakmals12/AI-Project-Model-Convolutional-Neural-Network-for-Early-Detection-of-Chili-Plant-Diseases/assets/72428679/33d5a326-e8ad-4841-8db3-5a55e24ce413)


#### B. Training Result
The training results were obtained from experiments that were carried out using 100 epochs, 32 batch sizes, loss functions using categorical cross-entropy, and Adam's optimizer. Based on the graphic below, which shows the training results of the two graphs from the training process using CNN and DenseNet201, then for the model. fit configuration, namely training data using train-ing_set and validation data using test_set there is no overfitting. The training process uses Google Collaboratory using Google's GPU run time so that the CNN and DenseNet201 model training process can be done faster. This figure shows the results of the training using the two models a,b CNN model, c,d DenseNet201 model.

![image](https://github.com/Rifqiakmals12/AI-Project-Model-Convolutional-Neural-Network-for-Early-Detection-of-Chili-Plant-Diseases/assets/72428679/b7aeb83e-200a-45c3-b489-f1f6d5f40b95)

Compare the two models and it can be done by looking at the validation values of each of these models. The CNN model trained for 100 epochs results from the smallest validation loss value at the 90th epoch with a validation loss value of 0.2311. Then for the DenseNet201 model, which has the smallest process of 100 epochs, the results of the validation loss value are obtained at the 20 epochs with a loss validation value of 0.2507. The value of validation loss at the training stage is shown in the table.

![image](https://github.com/Rifqiakmals12/AI-Project-Model-Convolutional-Neural-Network-for-Early-Detection-of-Chili-Plant-Diseases/assets/72428679/bd954345-214c-477a-9652-efaa7b125f09)


#### C. Confusion Matrix Result 
To measure the CNN and DenseNet201 models that have been made, whether the performance of the model is good or not by using the confusion matrix method, where testing it uses testing data consisting of 5 healthy classes, leaf curl, leaf spot, whitefly, yellowish where each class has 36 images. The confusion matrix results from the CNN and DenseNet201 models can be seen in the table below. In the CNN model, the results of image predictions were classified correctly 33 of 36 images were classified as "healthy", 36 of 36 images were successfully classified as "leaf curl", 33 of 36 images were successfully classified as "leaf spot", 35 of 36 images were successfully classified "whitefly", and 32 of the 36 images were successfully classified as "yellowish". Then for the DenseNet201 model, the results of the image prediction were correctly classified 34 out of 36 images were classified as "healthy", 32 out of 36 images were successfully classified as "leaf curl", 32 out of 36 images were successfully classified as "leaf spot", 35 out of 36 images were successfully classified "whitefly", and 32 of the 36 images were successfully classified as "yellowish". These tables show the confusion matrix results on the CNN and DenseNet201 models.

![image](https://github.com/Rifqiakmals12/AI-Project-Model-Convolutional-Neural-Network-for-Early-Detection-of-Chili-Plant-Diseases/assets/72428679/829f92ca-077b-4205-a6d2-7c945155d2f3)

Furthermore, after getting the confusion matrix can be used to calculate the value of precision, recall, f1-score, and accuracy. Based on the confusion matrix table for the CNN and DenseNet201 models above, the calculation results for the precision, recall, f1-score, and accuracy values of the CNN and DenseNet201 models using the classi-fication_report() function in the scikit learn library can be seen in the table below. Based on these results, the CNN model has a better accuracy rate of 94%. Then the DenseNet201 model produces an accuracy rate of 92%.

![image](https://github.com/Rifqiakmals12/AI-Project-Model-Convolutional-Neural-Network-for-Early-Detection-of-Chili-Plant-Diseases/assets/72428679/7c7d9458-9b3e-4853-93e1-1081eaa7eae0)

#### D. Testing Result 
The experiment uses random image data from the testing directory. The experimental and classification results using CNN can be seen in the figure. The experiment results display 15 randomly selected test images and have two outputs: the actual image class and class classification carried out by the previously trained CNN and DenseNet201 models. 

![image](https://github.com/Rifqiakmals12/AI-Project-Model-Convolutional-Neural-Network-for-Early-Detection-of-Chili-Plant-Diseases/assets/72428679/a569f9f9-db13-4c97-8ba2-935c8ff077d2)


## Conclusion
The experimental results of the CNN model obtained testing results with a success rate of 100%, where the CNN model managed to predict 15 of the 15 images. Then the DenseNet201 model obtained testing results with a success rate of 87%, where the DenseNet201 model predicted 13 of the 15 images correctly, and two images were mis predicted. Based on these results, the CNN model has a better level of accuracy than DenseNet201 in conducting the disease classification process in chili plants.

## Contact

If you have questions or feedback, feel free to reach out to us at [rifqias1212@gmail.com].

Thank you for exploring our project or portfolio!



