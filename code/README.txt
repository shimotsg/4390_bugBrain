## README ##
----------
12-21-2018

Geoffrey Shimotsu 
geoffrey.shimotsu01@utrgv.edu

This file details how to run code produced for the CSCI4930 project. All code run using Python 3.6.


----------


ensembleModel.py
----------------
This file requires file structuring: 

 - trainingImages
	 - trainingImagesClass1
	 - trainingImagesClass2
	 - ...
	 - trainingImagesClassN
 - validationImages
	 - validationImagesClass1
	 - validationImagesClass2
	 - ...
	 - validationImagesClassN

Specify the paths to training and validation folders in **lines 30 and 31**.

Paths to `model.state_dict` files for ResNet18 models (initialized with last fc layer reshaped) must be specified for initialized models in **lines 142, 156, & 170**.


gammaCorrect.py
-------
This script applies gamma correction and saves a corrected copy of all images in directory. Directory must only contain images. Change the gamma value on **line 16** to desired value, save and run script. Optionally change prefix on **line 28**.
imageScrape.py
-------
This script opens a Selenium browser and downloads Google image search results. Specify PATH for images on **line 12**. Specify the search term string on **line 14**. If >100 images desired, login to Google and modify url string on **line 15**. To prevent scrolling, modify second parameter in **line 27**. 
singleModelFeatureExtract.py
-------
This file requires file structuring: 

 - trainingImages
	 - trainingImagesClass1
	 - trainingImagesClass2
	 - ...
	 - trainingImagesClassN
 - validationImages
	 - validationImagesClass1
	 - validationImagesClass2
	 - ...
	 - validationImagesClassN

Specify PATHs to training and validation sets on **lines 38 and 39**.

singleModelFinetune.py
-------
This file requires file structuring: 

 - trainingImages
	 - trainingImagesClass1
	 - trainingImagesClass2
	 - ...
	 - trainingImagesClassN
 - validationImages
	 - validationImagesClass1
	 - validationImagesClass2
	 - ...
	 - validationImagesClassN

Specify names of training and validation set folders on **lines 48 and 49**. Specify finetuning or feature extraction with bool value on **line 45**. **Line 170** defines a PATH to save a `model.state_dict`.
