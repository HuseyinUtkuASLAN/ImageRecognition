# ImageRecognition
Project to find which pictures have human in them. For feature extraction Histogram of Gradient (HOG) is used and for classification vector machine, k-means clustering and feed forward neural network is used.

## Overview
Purpose of this project is detecting pictures which has humans in it and hasn’t. Name of the application is “bye cruel world”. There are no particular reason for me choosing it.
## Steps
1. Extract features with Histogram of Gradients method. 
2. Flatten the data. 
3. Train classifier or create clusters. 
4. Test and compare the results.
## Specifications
To run this program user needs some dependencies installed on their machine. These are 
python3, python3-opencv, python3-numpy and python-sklearn.  <br /> 
Images should be stored in “Images” file. Inside that folder there are 2 others which represents 
classes; positives and negatives as “pos” and “neg” To run the program user should enter 
“python main.py -e True -c SVM” to terminal. Script works both with python 2 and 3. If user wants 
to run feature extraction before classification, user needs to enter “-e True” . If not, “-e False” 
should be entered. At the end of extraction, program saves the flattened data to a “.npy” file. 
After running it once, program will use saved data for training classifier. There are 2 classification 
and 1 clustering method. These are Support Vector Machine(SVM), neural network and k-means 
algorithms. To run SVM, user needs to enter “-c SVM”. To run neural network, user needs to enter 
“-c MLP”. To run k-means, user needs to enter “-c kmeans”.  
<br/>
Examples to run :
```
python main.py -e True -c SVM 
python main.py -e False -c MLP 
python main.py -e True -c kmeans 
```
Requirements  :
```
python, numpy, OpenCV, sklearn, scipy, matplotlib, imutils
```
More can be found in documentation.<br/>
Learning and testing data is taken from INRIA data set. [[link]](http://pascal.inrialpes.fr/data/human/) 
