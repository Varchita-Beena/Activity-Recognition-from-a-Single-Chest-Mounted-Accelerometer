import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
column_names = ['seq','x','y','z','label']

#load subject data to asngle array
def get_data(start, stop):
	total_data = []
	for i in range(start,stop):
		file = str(i)+'.csv'
		data = pd.read_csv(file,index_col=False,names=column_names)
		data = data.drop(['seq'], axis=1)
		data = data[data.label != 0]
		total_data.append(data.values)
	total_data = np.array(total_data)
	return total_data
array = get_data(1,16) #load all 15 subjects' data to single array
print("total_datasets",array.shape) #shape(15,)

#plot x,y,z and activity for singles subject
def plot_subjects(subjects):
	plt.figure()
	# create a plot for each subject
	for i in range(len(subjects)):
		plt.subplot(len(subjects), 1, i+1)
		# plot each of x, y and z
		for j in range(subjects[i].shape[1]-1):
			plt.plot(subjects[i][:,j])
	plt.show()
plot_subjects(array[0:5])

#grouping data by activities
def group_by_activity(array, activities):
	grouped = [{a:s[s[:,-1]==a] for a in activities} for s in array]
	return grouped
def plot_durations(grouped, activities):
	freq = 52
	durations = [[len(s[a])/freq for s in grouped]for a in activities]
	plt.boxplot(durations, labels = activities)
	plt.show()
	
activities = [i for i in range(1,8)]
grouped = group_by_activity(array, activities)

#ploting durations
plot_durations(grouped, activities)

def histogram_subjects(subjects):
	plt.figure()
	# create a plot for each subject
	xaxis = None
	for i in range(len(subjects)):
		ax = plt.subplot(len(subjects), 1, i+1, sharex=xaxis)
		if i == 0:
			xaxis = ax
		# plot a histogram of x data
		for j in range(subjects[i].shape[1]-1):
			plt.hist(subjects[i][:,j], bins=100)
	plt.show()
histogram_subjects(array)

