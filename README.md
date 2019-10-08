# **Traffic Sign Recognition** 

In this project, I will use deep neural networks and convolutional neural networks to classify traffic signs. We will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.


---

**Build a Traffic Sign Recognition Project**

The steps of this project are the following:
* Load the data set 
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/1.png "Visualization"
[image2]:  ./examples/2.png "The characteristics of the images"
[image3]: ./examples/3.png "Distribuation"
[image4]: ./examples/4.png "Augmentation Methods"

[image5]: ./examples/S5.png "Traffic Sign 1"
[image6]: ./examples/S6.png "Traffic Sign 2"
[image7]: ./examples/S7.png "Traffic Sign 3"
[image8]: ./examples/S8.png "Traffic Sign 4"
[image9]: ./examples/S9.png "Traffic Sign 5"
[image10]: ./examples/S10.png "Traffic Sign 6"

[image11]: ./examples/T-00_2.jpg "Traffic Sign 7"
[image12]: ./examples/T_00_1.jpg "Traffic Sign 8"
[image13]: ./examples/T_01_1.jpg "Traffic Sign 9"
[image14]: ./examples/T_01_3.jpg "Traffic Sign 10"
[image15]: ./examples/T_23_1.jpg "Traffic Sign 11"
[image16]: ./examples/T_25_5.jpg "Traffic Sign 12"
[image17]: ./examples/T_27_1.jpg "Traffic Sign 13"
[image18]: ./examples/T_28_0.jpg "Traffic Sign 14"

[image19]: ./examples/19.png "The top five softmax probabilities for the clear images"
[image20]: ./examples/20.png "The top five softmax probabilities for the noisy images"

[image21]: ./examples/21.png  "Feature maps"



## Rubric Points

### Here I will describe how I addressed each step in my implementation.  


#### 1. Because our dataset is a To Load the data set Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.



After loading the dataset, I asserted weathers the numbers of the training images and labels are equal or not.

    assert(len(x_train)==len(y_train))
    assert(len(x_valid)==len(y_valid))
    assert(len(x_test)==len(y_test))



Then I got the following summary information:

    Number of training examples : 34799
    Number of testing examples : 12630
    Number of validation examples : 4410
    Image shape is: (32 32, 3)
    Number of classes labels : 43

After the summary section I read and printed the segmentation.csv file with help of the panadas library.

    ClassId 	SignName
    0 	Speed limit (20km/h)
    1 	Speed limit (30km/h)
    2 	Speed limit (50km/h)
    3 	Speed limit (60km/h)
    4 	Speed limit (70km/h)
    


#### 2. Include an exploratory visualization of the dataset.

Here is shown some images from the dataset with the coressponding titles:
![alt text][image1]




I was encouraged to to print several images for each label and try to  pay attention how the images look like. This is important because I need to know the characteristics of the images that you are using for training my model. 
![alt text][image2]


Now I am going to explore the distribution and take look at the comparing distribution of  each classes(training ,validation,test).

![alt text][image3]



    
Then I used the pandas library to calculate summary statistics of the traffic signs data set:

                                The maximum sign                  Number                 the minumu sign       Number
                        
    Training data   End of no passing by vehicles over 3.5 metric ... 	2010              Speed limit (20km/h)   	180
    Validation data End of no passing by vehicles over 3.5 metric ... 	240               Speed limit (20km/h)   	30  
    Test data       End of no passing by vehicles over 3.5 metric ... 	750               Speed limit (20km/h)   	60  
   



### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


From the comparing histograms, we can see that both training set and validation set have similar distribution of traffic sign samples. Problem is that there is a huge variability of the distribution between class instances within the dataset,maybe we can develop augmentation techniques to equalize them.
Because of inblancing all dataset I wanted to use augmentation methods like salt papper noise, rotation and translation methods
![alt text][image4]

There are three common forms of data preprocessing:

1-Mean subtraction is the most common form of preprocessing. It involves subtracting the mean across every individual feature in the data, and has the geometric interpretation of centering the cloud of data around the origin along every dimension.

2-Normalization refers to normalizing the data dimensions so that they are of approximately the same scale. 

3-PCA and Whitening is another form of preprocessing. In this process, the data is first centered as described above. Then, we can compute the covariance matrix that tells us about the correlation structure in the data.

 Befor the nomalization all images I converted all images to the LAB color system(L for lightness and a and b for the color opponents green–red and blue–yellow) to improve the contrast of my images by using CLAHE (Contrast Limited Adaptive Histogram Equalization) from the opencv librarythen (to only Lightness component and convert back the image to RGB) then I normalized so that the data has mean zero and equal variance.



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The process I did to find my solution was related to the accuracy of the model:

As known, one of the best archituctue in the fied of deep learing that has achieved much attention is the inception module. for this reason I used the inception modules to increase the accuracy of my model.


For visualizing the model architecture, I tried to open the tensorboard environment in the Udacity's workspace (tensorboard --logdir=logs) but I got many errors.

    Layer 	Description 	Input 	Output
    Inception with batch_normalization and relu  activation 	Inception3a 	(?,32,32,3) 	(?, 32, 32, 256)
    Inception with batch_normalization and relu  activation 	Inception3b 	(?,32,32,256) 	(?, 32, 32, 480)
    Max pooling 	 kernel: 2x2; stride:2x2; padding: Same;                	(?, 32, 32, 480) 	(?, 16, 16, 480)
    Inception with batch_normalization and relu  activation 	Inception3c 	(?, 16, 16, 480) 	(?, 16, 16, 766)
    Max pooling 	 kernel: 2x2; stride:2x2; padding: Same;                	(?, 16, 16, 766) 	(?, 8, 8, 766)
    Flatten 	Squeeze the cube into one dimension 	(?,8,8,766) 	(?,49024)
    Fully connected with dropout 	scope:fully_1; pairwise connections between all nodes 	(?,49024) 	(?,1024)
    Fully connected with dropout	scope:fully_2; pairwise connections between all nodes 	(?,1024) 	(?,512)
    Fully connected with dropout	scope:fully_3; pairwise connections between all nodes 	(?,512) 	(?,128)
    Fully connected with dropout	scope=out; pairwise connections between all nodes 	(?,128) 	(?,43)




#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model,I chose these hyperparameters based my experiences that I had with the taining phase. I tried to train my model for more epochs to see if I get a better result but I relized that a batch size of 256 can lead to a faster convergence.


Hyperparameter tuning

    LEARNING RATE = 0.0003
    EPOCHS = 5
    BATCH SIZE = 256
    Dropout keep probability rate : 0.5

Optimizer

     I chosed Adam opzimizer Adam (Adaptive Moment Estimation), In this algorithm, we divide the learning rate for a weight by a running average of the magnitudes of recent gradients for that weight. This helps in faster gradient descent and it is more accurate than SGD and GD
      
Then I passed the training data through the training pipeline to train the model.

Before each epoch, shuffle the training set.

After each epoch, measure the loss and accuracy of the validation set.

Save the model after training.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

     
 1-I tried data augmentation  which didn't help me 
 2-I Added dropout regularization at the end of each fully connected layer and achieved improvements, 
 3-Batch normalization is used after the first convolutional layer.
 
After 5 epochs I got a validation accuracy  1.0  and Loss of  0.03 and my final model was constructed, It took me about quarter  hour  to train on 5 iterations. After quarter hour, I got about  1.0   accuracy on the validation set with learning_rate=0.0003

The final results are:

    Train Accuracy 0.999655162505
    Validation Accuracy 1.04489795918
    Test Accuracy 1.01322248614
    
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image5] ![alt text][image7] ![alt text][image9] 
![alt text][image6] ![alt text][image8] ![alt text][image10]
 
 Here are six noisy traffic signs that I found on the web:

![alt text][image11] ![alt text][image12] ![alt text][image13] 
![alt text][image14] ![alt text][image15] ![alt text][image16]
![alt text][image17] ![alt text][image18]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.
Here are the result Indices of my predictions:
The top five softmax probabilities of the predictions on the captured images are outputted.


    ------------------------------------------------------------
    The label of the image:0  is 30 which has the corresponding predictions of  [[28 20 30  6 11]] 
    ------------------------------------------------------------
    The label of the image:1  is 1 which has the corresponding predictions of  [[28 20 30  6 11]] 
    ------------------------------------------------------------
    The label of the image:2  is 31 which has the corresponding predictions of  [[28 20 30  6 11]] 
    ------------------------------------------------------------
    The label of the image:3  is 29 which has the corresponding predictions of  [[28 20 30  6 11]] 
    ------------------------------------------------------------
    The label of the image:4  is 11 which has the corresponding predictions of  [[28 20 30  6 11]] 
    ------------------------------------------------------------
    The label of the image:5  is 8 which has the corresponding predictions of  [[28 20 30  6 11]] 

    
    for nisy images
    
    ------------------------------------------------------------
    The label of the image:0  is Pedestrians which has the corresponding predictions of  [[13  0  1  2  3]] 
    ------------------------------------------------------------
    The label of the image:1  is Speed limit (30km/h) which has the corresponding predictions of  [[13  0  1  2  3]] 
    ------------------------------------------------------------
    The label of the image:2  is Road work which has the corresponding predictions of  [[13  0  1  2  3]] 
    ------------------------------------------------------------
    The label of the image:3  is Speed limit (30km/h) which has the corresponding predictions of  [[13  0  1  2  3]] 
    ------------------------------------------------------------
    The label of the image:4  is Children crossing which has the corresponding predictions of  [[13  0  1  2  3]] 
    ------------------------------------------------------------
    The label of the image:5  is Slippery road which has the corresponding predictions of  [[13  0  1  2  3]] 
    ------------------------------------------------------------
    The label of the image:6  is Speed limit (20km/h) which has the corresponding predictions of  [[13  0  1  2  3]] 
    ------------------------------------------------------------
    The label of the image:7  is Speed limit (20km/h) which has the corresponding predictions of  [[13  0  1  2  3]] 






#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Predict the Sign Type for the clear Images!
[alt text][image19]


Predict the Sign Type for the noisy Images!
[alt text][image20]



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
Provided the function code that allows us to get the visualization output of any Tensorflow weight layer we want. The inputs to the function should be a stimuli image, one used during training or a new one you provided, and then the Tensorflow variable name that represents the layer's state during the training process.The result of the first cinvolutional layer is presented below.
![alt text][image21]



#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


