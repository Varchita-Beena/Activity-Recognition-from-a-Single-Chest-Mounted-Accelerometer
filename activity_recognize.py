import numpy as np
import random
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import skew, kurtosis, iqr, median_absolute_deviation
import pywt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

column_names = ['seq','x','y','z','label']#name of the columns

def get_data(start, stop):
	total_data = []
	for i in range(start,stop):
		file = str(i)+'.csv'
		data = pd.read_csv(file,index_col=False,names=column_names)
		data = data.drop(['seq'], axis=1)#dropping column having label seq
		data = data[data.label != 0]#dropping rows with label 0,2,6
		data = data[data.label != 2]
		data = data[data.label != 6]
		total_data.append(data)
	return total_data
	
def get_seperate_files(test_files):
	temp = random.sample(range(1, 16), test_files)
	total_data_test = []
	total_data_train = []
	for i in range(1,16):
		if i in temp:
			total_data = get_data(i)
			total_data_test.append(total_data)
		else:
			total_data = get_data(i)
			total_data_train.append(total_data)
	frame = pd.concat(total_data_train, axis=0, ignore_index=True)
	test_frame = pd.concat(total_data_test, axis=0, ignore_index=True)
	print('train data information by class')
	print(frame.groupby('label').count())
	print('test data information by class')
	print(test_frame.groupby('label').count())
	return frame, test_frame
	
def normalizing(data):
	scaled_list = data.tolist()
	list_data = min_max_scalar.transform(scaled_list)
	return list_data

def making_seq(list_data, labels_list):
	freq = 52
	train_sequences = []
	seq_class = []
	stop = len(list_data)
	for i in range(0, stop, 30):
		seq = list_data[i : i+freq]
		if len(seq) != freq:
			num = freq - len(seq)
			seq = seq + [[0.0,0.0,0.0]] * num
		train_sequences.append(seq)
		l = labels_list[i : i+freq]
		counts = Counter(l)
		max_key = max(counts, key = counts.get)
		seq_class.append(max_key)		
	train_sequences = np.array(train_sequences)
	return train_sequences, seq_class

def get_all_data():
	total_data = get_data(1,16)
	frame = pd.concat(total_data, axis=0, ignore_index=True)
	return frame
	
def get_labels_data(frame):
	labels = frame[frame.columns[-1]]	
	labels_list = labels.values.tolist()	#taking last column for label
	data = frame.drop(frame.columns[-1], axis=1)	#rest columns for data
	return labels_list,data
	
def cal_sq_root(a):
	sq = np.square(a)
	root = np.sqrt(np.sum(sq))
	return root

def extract_features(train_sequences):
	root = np.apply_along_axis(cal_sq_root, 2, train_sequences)
	train_sequences = np.insert(train_sequences,-1,root,axis = 2)	#add m dimension m = sqrt(x^2, y^2, z^2)
	frequency_domain = np.fft.fft(train_sequences, axis=1)	#changing in to frequency domain
	frequency_domain = np.absolute(frequency_domain)	#taking absolute to remove complex numbers
	#features from frequency_domain
	kur = kurtosis(frequency_domain, axis = 1)			#kutosis 
	integral = np.trapz(frequency_domain, axis = 1)		#taking integration (trapezodial) 
	skewness = skew(frequency_domain, axis = 1)			#skewness
	min_fd = np.min(frequency_domain, axis = 1)			#minimum
	max_fd = np.max(frequency_domain, axis = 1)			#maximum
	min_max_sum_fd = np.sum([min_fd, max_fd],axis= 0)	#minimum maximum sum
	var_fd = np.var(frequency_domain, axis=1)			#variance
	mean_fd  = np.mean(frequency_domain, axis=1)		#mean
	min_max_sub_fd = np.subtract(max_fd,min_fd)			#minimum maximum subtract
	#features from time_domain
	var= np.var(train_sequences, axis=1)				#variance
	mean = np.mean(train_sequences, axis=1)				#mean
	min = np.min(train_sequences, axis = 1)				#minimum
	max = np.max(train_sequences, axis = 1)				#maximum
	min_max_sum = np.sum([min, max],axis= 0)			#minimum maximum sum
	qr = iqr(train_sequences, axis = 1)					#inter quartile range
	mad = median_absolute_deviation(train_sequences, axis = 1)#mean absolute deviation
	min_max_sub = np.subtract(max,min)					#minimum maximum subtract
	
	feature = np.concatenate((var,mean,min,max,min_max_sum, qr, mad, min_max_sub, kur, integral, skewness, min_fd, max_fd, min_max_sum_fd, var_fd, mean_fd, min_max_sub_fd), axis=1)							#concat features
	return feature

def add_feature_names(all_features, dimension_names):
	feature_names = []	#giving features names along axis
	str = '_'
	for f in all_features:
		for dim in dimension_names:
			temp = f+str+dim
			feature_names.append(temp)
	return feature_names

def make_predictions(model,feature,class_label):
	pred = model.predict(feature)
	print('accuracy',accuracy_score(class_label, pred))
	print('total correctly classified',accuracy_score(class_label, pred, normalize = False))
	res = confusion_matrix(class_label, pred)
	print('confusion_matrix\n',res)
	normalized_res_train = res.astype('float')/ res.sum(axis = 1)[:,np.newaxis]
	print('normalized_rounded_confusion_matrix')
	rounded_train = np.round(normalized_res_train,2)
	print(rounded_train * 100)

	fig, ax = plt.subplots(figsize=(10,10))
	sns.heatmap(normalized_res_train, annot=True, fmt='.2f', xticklabels=[1,3,4,5,7], yticklabels=[1,3,4,5,7])
	plt.ylabel('Actual')
	plt.xlabel('Predicted')
	plt.show(block=False)
	plt.show()
	

frame = get_all_data()		# get data from all files (case 1)
'''
num_of_file_test = 3		#define how many files data for test , remaining will be for train (case 2)
frame, test_frame = get_seperate_files(num_of_file_test)
'''
print('train data info by class')
print(frame.groupby('label').count())
labels_list, data = get_labels_data(frame)
#test_labels_list, test_data = get_labels_data(test_frame) #for case 2 uncomment this 

#dividing data in sequences as per 52 Hz, each list = 52 sammples with approx 50% offset (sliding window)
train_sequences, seq_class = making_seq(data.values.tolist(), labels_list)
#test_sequences, test_seq_class = making_seq(test_data.values.tolist(), test_labels_list) #for case 2 uncomment this
train_sequences, test_sequences, ytrain, ytest = train_test_split(train_sequences, seq_class,stratify = seq_class) #test_train data divide with 25 % in test, for case 2 comment this

print("train_sequences",train_sequences.shape, 'test_sequences',test_sequences.shape)
#label encoder mapping [1,3,4,5,7] -> [0,1,2,3,4]
le = LabelEncoder()
le.fit(ytrain)
class_label = le.transform(ytrain)
test_class_label = le.transform(ytest) #for case 2 replace ytrain -> seq_class, ytest -> test_seq_class

print("train features\n")
feature = extract_features(train_sequences)

print("test features\n")
feature_test = extract_features(test_sequences)

dimension_names = ['x','y','z','m'] #total dimensions in our data after extracting features
all_features = ['variance','mean','minimum','maximum','min_max_sum','inter_quartile_range''mean_abs_deviation','min_max_subtract','kurtosis','trapazodial_integral','skewness','minimum_frequency_domain','maximum_frequency_domain','mini_max_sum_frequency_domain','variance_frequency_domain','mean_frequency_domain','min_max_subtract_frequency_domain']
feature_names_dimensions = add_feature_names(all_features, dimension_names)

print("normalize\n")
min_max_scalar = MinMaxScaler()
min_max_scalar.fit(feature)

print("train normalize\n")
feature = normalizing(feature)

print("test normalize\n")
feature_test = normalizing(feature_test)

print("train fit\n")
model = RandomForestClassifier(n_estimators = 100,class_weight = 'balanced',min_samples_leaf = 50)
model.fit(feature,class_label)

print("predictions\n")
print("On train data\n")
make_predictions(model,feature,class_label)
print("on test data\n")
make_predictions(model,feature_test,test_class_label)

print('feature name','importance')
for importance, name in sorted(zip(model.feature_importances_, feature_names_dimensions),reverse=True):
	print (name, importance)
