import argparse
from decimal import Decimal, getcontext

from imutils import paths
import imutils
import cv2
import numpy as np

from sklearn.svm import SVC

from hog import hog
from grayscale import rgb2grey

from kmeans import kmeans
from kmeans import plot

from neural_network import Dropout

# returns numpy array which populated with HOG valeus of images in 'path' folder
def hog_values(path,resize = False):
	array = []

	error_count = 0

	for imagePath in paths.list_images(path):
	
		image = cv2.imread(imagePath)
		if(resize):
			image = cv2.resize(image, (160, 96)) 
			
		gray_image = rgb2grey(image)

		try:
			array.append(hog(gray_image, orientations=9, pixels_per_cell=(32, 32),
	        	cells_per_block=(3, 3), normalise=False))
		except ValueError:
			print("ValueError : ",imagePath)
			error_count += 1
			continue 

	if(error_count != 0):
		print("\nError count : ", error_count)
	return np.array(array)

# < READ ARGUMENTS

feature_extract = True
feature_load = True

ap = argparse.ArgumentParser()
ap.add_argument("-e","--extract",required = True, help = "do you want to extract?")
ap.add_argument("-c","--classifier",required = True, help = "which classifier/clustering do you want to use? (SVM,kmeans)")
args = vars(ap.parse_args())

if args["classifier"] == "SVM":
	classifier = "SVM"
elif args["classifier"] == "MLP":
	classifier = "MLP"
elif args["classifier"] == "kmeans":
	classifier = "kmeans"
else:
	print("There is no classifier/clustering named that way")
	quit()


if args["extract"] == "True":
	feature_extract = True
else:
	feature_extract = False

# /READ ARGUMENTS >

testPositives = []
testNegatives = []
trainPositives = []
trainNegatives = []


print("\n==========HOG==========\n")

# if extract argument is given by user, extracts HOG values and saves them on npy files
if(feature_extract):
	print("\n-----------------------\n")
	print("\tExtracting features")

	path_positive = "./Images/pos"
	path_negative = "./Images/neg"

	print("\n\tHOG Positives")
	positives = hog_values(path_positive,True) # positives for this examples dont need resizing
	print("\t",positives.shape, " positives.")

	print("\n\tHOG Negatives")
	negatives = hog_values(path_negative,True)
	# print(negatives)
	print("\t",negatives.shape, " negatives.")


	np.save("positive",positives)
	print("\tPositives HOG vectors are saved")
	np.save("negatives",negatives)
	print("\tNegatives HOG vectors are saved")
	print("\n-----------------------\n\n")


hard_example = hog_values("./Images/hard_examples",True)# result should be 2
hard_example1 = hog_values("./Images/hard_examples1",True)# result should be 1
hard_example2 = hog_values("./Images/hard_examples2",True)# result should be 1
hard_example3 = hog_values("./Images/hard_examples3",True)# result should be 2

# loads numpy arrays from "positive.npy" and "nehatives.npy"
if(feature_load):
	print("Loading features : ")
	positives = np.load("positive.npy")
	negatives = np.load("negatives.npy")



	testPositives = positives[:300]
	testNegatives = negatives[:300]
	trainPositives = positives[:-300]
	trainNegatives = negatives[:-300]

	print("positives : ", trainPositives.shape, testPositives.shape)
	print("negatives : ", trainNegatives.shape, testNegatives.shape)

trainData = np.concatenate((trainPositives,trainNegatives), axis=0)
testData = np.concatenate((testPositives,testNegatives), axis=0)

# uses sklearn's SVM to compare results with other classification/clustering
if classifier == "SVM":
	
	ones = np.ones([trainPositives.shape[0],1])
	twos = np.ones([trainNegatives.shape[0],1]) * 2

	classes = np.concatenate((ones,twos), axis=0)

	print("\nTraining data : ", trainData.shape) 
	print("Test data : ", testData.shape)
	clf = SVC(gamma = 0.1, C = 100)



	print("\n",clf.fit(trainData,(classes.T)[0]))



	error = 0

	for row in testPositives:
		if(clf.predict([row]) != 1):
			error += 1

	for row in testNegatives:
		if(clf.predict([row]) != 2):
			error += 1

	print("\nError Rate : ", error , "/", len(testPositives) + len(testNegatives) )

	print("\nHard Example 0: ", clf.predict(hard_example)) # result should be 2
	print("\nHard Example 1: ", clf.predict(hard_example1)) # result should be 1
	print("\nHard Example 2: ", clf.predict(hard_example2)) # result should be 1
	print("\nHard Example 3: ", clf.predict(hard_example3)) # result should be 2

# uses kmeans clustering algorithm with k = 2
# even though this is an unsupervised learning algorithm, we know that there are only 2 classes
# for each test data, we test wether or not nodes with same classes are in the same cluster
if classifier == "kmeans":

	data_plot = False

	print("\n--------kMeans--------\n")

	centroids, history_centroids, belongs_to = kmeans(2,trainData)
	# print(centroids)
	if data_plot:
		plot(trainData, history_centroids, belongs_to)

	zeros = 0
	ones = 0

	for v in testPositives:
		dist0 = np.linalg.norm(v-centroids[0])
		dist1 = np.linalg.norm(v-centroids[1])
		if dist0 < dist1:
			ones += 1
		else:
			zeros += 1
	print("-----Class1-----")
	print("Number of positives in cluster 0 : ", zeros)
	print("Number of positives in cluster 1 :", ones)
	zeros = 0
	ones = 0

	for v in testNegatives:
		dist0 = np.linalg.norm(v-centroids[0])
		dist1 = np.linalg.norm(v-centroids[1])
		if dist0 < dist1:
			ones += 1
		else:
			zeros += 1
	print("-----Class2-----")
	print("Number of positives in cluster 0 : ", zeros)
	print("Number of positives in cluster 1 :", ones)

	hard_examples = np.concatenate((hard_example, hard_example1,hard_example2,hard_example3),axis = 0)
	for i,v in enumerate(hard_examples):
		dist0 = np.linalg.norm(v-centroids[0])
		dist1 = np.linalg.norm(v-centroids[1])
		if dist0 < dist1:
			print("\nHard Example ",i,": ", 1)
		else:
			print("\nHard Example ",i,": ", 2)



# a multilayer perceptron with dropout for generalization and relu for activation
# due to distinct attributes of data, i had to implement a practical solution
# because of limits of float variables, we can not calculate exp(float) for too long
# instead of running training method 100000 times, algorithm runs it for 375 times with 400 examples for each class
# at the end of prediction, output with highest value is accepted as the predicted value
if classifier == "MLP":
	
	getcontext().prec = 10

	ones = []
	twos = []
	for i,tp in enumerate(trainPositives[:400]):
		ones.append([1,0])

	ones = np.array(ones)

	for i,tp in enumerate(trainNegatives[:400]):
		twos.append([0,1])

	twos = np.array(twos)

	classes = np.concatenate((ones,twos), axis=0)

	numInputNeuron = trainPositives.shape[1]
	## number of neurons in hidden layer (bias is not included)
	numHiddenNeuron = int(trainPositives.shape[1] / 2)
	## number of neurons in output layer
	numOutputNeuron = 2
	trainPositives = trainPositives[:400]
	trainNegatives = trainNegatives[:400]
	trainData = np.concatenate((trainPositives,trainNegatives), axis=0)
	trainData = trainData / np.linalg.norm(trainData)
	# trainNegatives = trainNegatives / np.linalg.norm(trainNegatives)
	print("\nTraining ...")
	
	nn = Dropout(trainData,classes,243,[400,400],2)
	nn.train(epochs = 375,lr = 0.01)

	print("Training ended\n")

	numberOfTests = 50

	result_negative = nn.predict(testNegatives[:numberOfTests])
	result_positive = nn.predict(testPositives[:numberOfTests])

	falseNegative = 0
	falsePositive = 0
	for r in result_negative:
		if(r[1] < r[0]):
			falseNegative += 1


	for r in result_positive:
		if(r[1] > r[0]):
			falsePositive += 1


	print("False negative : ",falseNegative,"/", numberOfTests,"\nFalse positive : ",falsePositive , "/",numberOfTests)

	hard_examples = np.concatenate((hard_example, hard_example1,hard_example2,hard_example3),axis = 0)

	hard_solutions = nn.predict(hard_examples)

	for i,v in enumerate(hard_solutions):
		if(v[1] < v[0]):
			print("\nHard Example ",i,": positive")
		else:
			print("\nHard Example ",i,": negative")

	print(hard_solutions)

		



	
print("\n=======================\n")

print("bye cruel world")