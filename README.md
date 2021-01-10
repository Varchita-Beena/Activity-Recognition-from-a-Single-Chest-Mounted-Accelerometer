# Activity-Recognition-from-a-Single-Chest-Mounted-Accelerometer
Problem of predicting or classifying what a person is doing based on a trace of their movement using sensors.

Human Activity Recognition(HAR), is the problem of predicting what a person is doing based on a trace of their movement using sensors.<br/>
Movements are often normal indoor activities such as standing, sitting, jumping, and going up stairs. Sensors are often located on the subject such as a smartphone or vest and often record accelerometer data in three dimensions (x, y, z).<br/>
To classify the activities and predicting the activites on upcomimg motions.<br><br/>
## Dataset<br/>
It is freely available from the UCI Machine Learning repository under the name of <br/>
Activity Recognition from Single Chest-Mounted Accelerometer Data Set<br/>
The dataset collects data from a wearable accelerometer mounted on the chest. Uncalibrated Accelerometer Data are collected from 15 participants performing 7 activities. The dataset is intended for Activity Recognition research purposes. It provides challenges for identification and authentication of people using motion patterns.<br/><br/>
## Dataset Information<br/>
Data are separated by participant<br/>
Each file contains the following information:<br/>
sequential number, x acceleration, y acceleration, z acceleration, label<br/>
Labels are codified by numbers<br/>
1: Working at Computer<br/>
2: Standing Up, Walking and Going up\down stairs<br/>
3: Standing<br/>
4: Walking<br/>
5: Going Up\Down Stairs<br/>
6: Walking and Talking with Someone<br/>
7: Talking while Standing<br/>
<br/>
<br/>
For plots use file : plot_traces.py<br/>
For model use file : activity_recognize.py<br/>
### plots from data uisng plot_traces.py<br/>
##### Plot for x acceleration for subject 7
### ![Plot for x acceleration for subject 7](https://github.com/Varchita-Beena/Activity-Recognition-from-a-Single-Chest-Mounted-Accelerometer/blob/main/subject_7_x.png)
##### Plot for y acceleration for subject 7
### ![Plot for y acceleration for subject 7](https://github.com/Varchita-Beena/Activity-Recognition-from-a-Single-Chest-Mounted-Accelerometer/blob/main/subject_7_y.png)
##### Plot for z acceleration for subject 7
### ![Plot for z acceleration for subject 7](https://github.com/Varchita-Beena/Activity-Recognition-from-a-Single-Chest-Mounted-Accelerometer/blob/main/subject_7_z.png)
We can also see that some activities are performed for much longer than others. This may impact the ability of a model to discriminate between the activities, e.g. activity 3 (standing) for both subjects has very little data relative to the other activities performed.<br/>
##### plot for activity for subject 7
### ![plot for activity for subject 7](https://github.com/Varchita-Beena/Activity-Recognition-from-a-Single-Chest-Mounted-Accelerometer/blob/main/subject_7_activity.png)
##### plot for all subject together
### ![plot for all subject together](https://github.com/Varchita-Beena/Activity-Recognition-from-a-Single-Chest-Mounted-Accelerometer/blob/main/plot_trace_each_subject_together.png)
Looking for general trend across the traces.<br/>
We can see that each subject has the same large spikes in the trace in the beginning of the sequence (first 60 seconds), perhaps related to the start-up of the experiment.<br/>
We can see lots of orange and green and very little blue, suggesting that perhaps the z data is less important in modeling this problem.<br/>
##### plot for histogram for all subjects
### ![plot for histogram for all subjects](https://github.com/Varchita-Beena/Activity-Recognition-from-a-Single-Chest-Mounted-Accelerometer/blob/main/histogram_all_subjects.png)
A histogram of the distribution of observations for each axis of accelerometer data.<br/>
The hist() function is used to create a histogram for each axis of accelerometer data, and a large number of bins (100) is used to help spread out the data in the plot.<br/>
The plot really helps to show both the relationship between the distributions within a subject and differences in the distributions between the subjects<br/>
This plot suggest that the distribution of each axis of accelerometer is Gaussian or really close to Gaussian.<br/>
Within each subject, a common pattern is for the x (blue) and z (green) are grouped together to the left and y data (orange) is separate to the right. The distribution of y is often sharper where as the distributions of x and z are flatter.<br/>
##### plot durations for all subjects
### ![plot durations for all subjects](https://github.com/Varchita-Beena/Activity-Recognition-from-a-Single-Chest-Mounted-Accelerometer/blob/main/plot_durations_all_subjects.png)
To find how long or how many observations we have for each activity across all of the subjects. This may be important if there is a lot more data for one activity than another, suggesting that less-well-represented activities may be harder to model.<br/>
We can see that there is relatively fewer observations for activities, 2 (standing up, walking and going up/down stairs), 5 (going up/down stairs) and 6 (walking and talking).<br/>
We can also see that each subject spent a lot of time on activity 1 (standing Up, walking and going up/down stairs) and activity 7 (talking while standing).<br/>

<br/>
<br/>

## Framing the data<br/>
Sliding Windows : The contiguous trace for each subject is split into sliding windows, with overlap (approx 50%) and the mode of each activity for a window is taken as the activity to be predicted.<br/>
Classes taken into consideration [1,3,4,5,7] (Working at Computer,Standing,Walking,Going Up\Down Stairs,Talking while Standing>)<br/>
<br/>
## Model<br/>
Random Forest model is used to train and test the data.<br/>
Why Random Forest<br/>
They are based on trees, so scaling of the variables doesn't matter. Any monotonic transformation of a single variable is implicitly captured by a tree.<br/>
They use the random subspace method and bagging to prevent overfitting.<br/>
If they are done well, you can have a random forest that deals with missing data easily.<br/>
Automated feature selection is built in.<br/>
<br/>
## Features Extracted
All the features are extracted along x,y,z,m accceleration<br/>
m is added to the dataset as: square root of addition of squares of x, y, z.
From time domain:<br/>
variance, mean, minimum, maximum , min_max_sum , inter_quartile_range, mean_abs_deviation , min_max_subtract.
From Frequency domain:<br/>
kurtosis, trapazodial_integral, skewness, minimum, maximum, mini_max_sum, variance, mean, min_max_subtract<br/>
Total 68 features (17 x 4)<br/>
<br/>
## Approaches<br/>
1. Taking all users : Combining all the data from all the users (from all 15) and divide them into train and test.<br/>
2. Divide by subjects : Take 12 to 13 users data in to train and test on remaining users<br/>
3. Single Subject : Divide train and test from single users<br/>
## Results<br/>
### Taking all Users (subjects) :<br/>
Train data has 45688 data points and test data has 15230 data points<br/>
##### Train Accuracy : 86.55%<br/>
##### Confusion Matrix for train data<br/>
![Confusion Matrix for train data](https://github.com/Varchita-Beena/Activity-Recognition-from-a-Single-Chest-Mounted-Accelerometer/blob/main/confusion_matrix_train.png)
##### Test Accuracy : 84.88%<br/>
##### Confusion Matrix for test data<br/>
![Confusion Matrix for test data](https://github.com/Varchita-Beena/Activity-Recognition-from-a-Single-Chest-Mounted-Accelerometer/blob/main/confusion_matrix_test.png)
<br/>
##### Total Features 68<br>
Here are some top features with their importance<br/>
maximum_frequency_domain_m : 0.0361020274197541<br>
min_max_subtract_frequency_domain_m : 0.03476062545751772<br>
mini_max_sum_frequency_domain_m : 0.03110345590560009<br>
variance_frequency_domain_m : 0.030267915786956905<br>
mean_m : 0.02961000635060823<br>
inter_quartile_range_y : 0.02952656595182603<br>
kurtosis_frequency_domain_y : 0.027024414800361194<br/>
trapazodial_integral_frequency_domain_y : 0.02652868865587016<br/>
variance_y : 0.02619098927283023<br/>
min_max_subtract_y : 0.024501741581540493<br/>

### Divide by Subjects :<br/>
Test files : 14, 7---Remaining files in Train<br/>
Train data points 51892<br/>
Test data points 9027<br/>
##### Train accuracy : 83.64 
##### Test accuracy : 64.29
There is a lot of varaince between subjects. So there is a big generalization gap.
##### Confusion Matrix Train
![Confusion Matrix Train](https://github.com/Varchita-Beena/Activity-Recognition-from-a-Single-Chest-Mounted-Accelerometer/blob/main/subject_divide_cm_train.png)
##### Confusion Matrix Test
![Confusion Matrix Test](https://github.com/Varchita-Beena/Activity-Recognition-from-a-Single-Chest-Mounted-Accelerometer/blob/main/subject_divide_cm_test.png)
### For Single Subject :
Train data points 3966<br/>
Test data points 1322<br/>
##### Train accuracy : 89.61 
##### Test accuracy : 87.98
##### Confusion Matrix Train
![Confusion Matrix Train](https://github.com/Varchita-Beena/Activity-Recognition-from-a-Single-Chest-Mounted-Accelerometer/blob/main/subject_1_CM_train.png)
##### Confusion Matrix Test
![Confusion Matrix Test](https://github.com/Varchita-Beena/Activity-Recognition-from-a-Single-Chest-Mounted-Accelerometer/blob/main/subject_1_CM_test.png)







