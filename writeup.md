# Traffic Sign Recognition

By Carlos Asmat 

[//]: # (Image References)

[image1]: ./img/output_16_1.png "Raw Data"
[image2]: ./img/output_26_0.png "Validation Accuracy by Epoch"
[image3]: ./img/output_31_3.png "Google Street-view Test Data Preprocessed"
[image4]: ./img/output_9_0.png "Classes Histogram"
[image5]: ./img/output_16_2.png "Preprocessed Data"
[image6]: ./img/output_31_2.png "Google Street-view Test Data"
[image7]: ./img/output_8_2.png "Training Data"

## Objective

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Data Set Summary & Exploration

### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

### Traffic Data Stats (as obtained from the shape of the arrays)

- Number of training examples = 34799
- Number of testing examples = 12630
- Image data shape = (32, 32, 3)
- Number of classes = 43
- Keys in train dataset:
  - coords: (34799, 4)
  - features: (34799, 32, 32, 3)
  - sizes: (34799, 2)
  - labels: (34799,)

#### 2. Include an exploratory visualization of the dataset.

The matrix below shows a sample of the traffic sign data in the training set.
In it, we notice the signs are always orthogonal to the camera and are cropped fairly evenly. There's a variety of, contrast, brightness and saturations.

![Training Data Sampler][image7]

The histogram below, shows the distribution of all the traffic sign according to their labels. Some are much more common than others, which probably reflects their frequency in real life. This means that some signs will be easier to train than others.

![Training Data Class Distribution][image4]

## Design and Test a Model Architecture

### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

As shown in the image below, the signs are not always located in the same image region and the contrasts and colours are not constant.

![Raw Training Data][image1]

Three steps of preprocessing were implemented: masking, grayscale conversion, contrast enhancement

#### 1 - Masking:
Since the training data includes coordinates for masking a region of interest, and in the hope to help the learning model, it seems reasonable to implement this effect.

Once this effect was applied however, it became obvious that the provided bounding boxes may do more harm than good, since very often they would mask useful regions of the signs. This would even make it difficult for a human to recognize some signs.

Although this effect is not used in the final model, it can be enabled by setting ```cropping = True``` in the accompanying Jupyter notebook.

#### 2 - Grayscale Conversion:
The low hanging fruit of preprocessing effects, this was quickly implemented in the hope it would remove unnecessary data from the training set.

It was found out however that the colour data is actually important for identifying the signs (unsurprisingly). 

Although this effect is not used in the final model, it can be enabled by setting ```gray = True``` in the accompanying Jupyter notebook.

#### 3 - Contrast Enhancement:
In order to make the training set more uniform, contrast enhancement was implemented for both colour images (as a Contrast Limited Adaptive Histogram Equalization or CLAHE), and grayscale (as a simple histogram equalization).

This enhances the features of the training set and makes it more uniform. THis effect has a positive impact in the accuracy of the final model. 

### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The final model used is LeNet adapted for colour images and 43 classes. It was cleverly renamed as LeTraffic.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling       	| 2x2 stride,  outputs 14x14x6					|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x16	|
| RELU					| 												|
| Max pooling       	| 2x2 stride,  outputs 5x5x16					|
| Flatten 				| outputs 400									|
| Fully connected		| outputs 120									|
| RELU					| 												|
| Fully connected		| outputs 84									|
| RELU					| 												|
| Fully connected		| outputs 43									|

### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The model was trained by using the *reduce mean* for the loss operation and the *Adams* optimizer.

The learning rate used is 0.001. More experimentation was done with lower and higher values, but 0.001 yielded the best result.  

The batch size of 128 also gave good results and any larger or smaller sizes would yield lower accuracy.

100 epochs were used in order to allow the model to attain a good level of accuracy.

### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The final model results are:
* training set accuracy of 1.00
* validation set accuracy of 0.947 
* test set accuracy of 0.918

In order to achieve this results, all combinations of preprocessing were tried as explained previously.

Also, the batch size were changes in the hope of accelerating the training process, however, the accuracy would be reduced by larger batch sizes.

The main problem encountered during the training was a mistake done in the loading of the data. The data was loaded as integers (0 to 255) instead of normalized floats (0 to 1). THis made the training converge with more difficulty.

Finally, the number of epochs was gradually increased from 10 to 100 in order to allow the model to reach a better accuracy.

Some adjustments to the layer sized was also done, but it did not help with the accuracy so they were reverted.

The robustness of the model to the different inputs (colour, grayscale, masked, equalized) shows the strength of the convolution layers that are able to extract features successfully in a wide range of situations.

Even with the large number of epochs, the LeTraffic (aka LeNet) model did not overfit and performed well in the test set.
 

## Test a Model on New Images

### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

13 traffic signs were extracted from browsing through the streets of Dusseldorf in Google Street-view.

The images were resized and cropped using Gimp.

![Traffic Signs from Dusseldorf][image6]

As shown in the visualization above, some of the images were purposefully chosen in a more challenging angle in order to test the model's robustness.

The same preprocessing was applied to these images.

![Traffic Signs from Dusseldorf Preprocessed][image3]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			  	        |     Prediction	        	| 
|:-------------------------:|:-----------------------------:| 
| Ahead only               	| Ahead only					|
| Ahead only               	| Ahead only					|
| Ahead only               	| Keep right					|
| Turn left ahead          	| Turn left ahead				|
| Turn right ahead         	| Turn right ahead				|
| Priority road            	| Priority road					|
| Yield                    	| Yield							|
| Roundabout mandatory     	| Yield							|
| No entry                 	| No entry						|
| Speed limit (50km/h)     	| Speed limit (50km/h)			|
| Speed limit (30km/h)     	| Speed limit (50km/h)			|
| Stop                     	| Stop							|
| Road work                	| Road work						|

10 Correct predictions out of 13

The model was able to correctly guess 10 of the 13 traffic signs, which gives an accuracy of 76.9%. This compares favourably to the accuracy on the test set of 91.8% especially since some signs were specifically chosen to be difficult to identify (and they were indeed misidentified)

### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 64th cell of the Ipython notebook.


| 35 | Label (Ahead only)						   | Probability 			|
|:--:|:-------------------------------------------:|:----------------------:| 
| 35 | Ahead only                                  | 1.0					|
| 40 | Roundabout mandatory                        | 1.5638148853522283e-15	|
| 16 | Vehicles over 3.5 metric tons prohibited    | 1.4171344688170133e-22	|
| 36 | Go straight or right                        | 1.827468546603392e-23	|
| 33 | Turn right ahead                            | 3.533127345337616e-24	|


| 35 | Label (Ahead only)						   | Probability 			|
|:--:|:-------------------------------------------:|:----------------------:| 
| 35 | Ahead only                                  | 1.0					|
| 40 | Roundabout mandatory                        | 5.2462514332773935e-09	|
| 16 | Vehicles over 3.5 metric tons prohibited    | 3.407236352444726e-17	|
| 37 | Go straight or left                         | 1.6169398326880978e-19	|
| 11 | Right-of-way at the next intersection       | 3.424127373282917e-22	|


| 38 | Label (Keep right)						   | Probability 			|
|:--:|:-------------------------------------------:|:----------------------:| 
| 35 | Ahead only                                  | 0.9831475615501404		|
| 5  | Speed limit (80km/h)                        | 0.007768301293253899	|
| 28 | Children crossing                           | 0.0077159409411251545	|
| 13 | Yield                                       | 0.0008942649001255631	|
| 24 | Road narrows on the right                   | 0.00029210231150500476	|


| 34 | Label (Turn left ahead)					   | Probability 			|
|:--:|:-------------------------------------------:|:----------------------:| 
| 34 | Turn left ahead                             | 0.9999902248382568		|
| 11 | Right-of-way at the next intersection       | 9.494656296737958e-06	|
| 40 | Roundabout mandatory                        | 2.710596049837477e-07	|
| 18 | General caution                             | 6.326978940762729e-09	|
| 35 | Ahead only                                  | 8.04869192128077e-12	|


| 33 | Label (Turn right ahead)					   | Probability 			|
|:--:|:-------------------------------------------:|:----------------------:| 
| 33 | Turn right ahead                            | 1.0					|
| 40 | Roundabout mandatory                        | 6.116116539190943e-17	|
| 11 | Right-of-way at the next intersection       | 1.1494554524800923e-18	|
| 35 | Ahead only                                  | 1.666113083226359e-24	|
| 13 | Yield                                       | 1.0704469747071413e-24	|

The rest of the probabilities for each image can be found in the Jupyter notebook.
