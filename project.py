'''
An implementation of LSTM in time-series forecasting
'''


'''
Data Preprocessing

preparing the data for supervised learning

'''
#returns a single record for datetime 
from datetime import datetime
def parse(x):
	return datetime.strptime(x, '%Y %m %d %H')

#read csv file into dataset as dataframe 
#and simultaneously merge all the date time columns
from pandas import read_csv
dataset = read_csv('beijing.csv',  parse_dates = [['year', 'month', 'day', 'hour']], 
                   index_col=0, date_parser=parse) #first column is the index

#drop first column, so the successive column becomes the index
dataset.drop('No', axis=1, inplace=True)

#set the name of the current index 
dataset.index.name = 'date'

#mark all NA values with 0
dataset['pm2.5'].fillna(0, inplace=True)

# drop the first 24 hours
dataset = dataset[24:]

#extract the values
values = dataset.values

#encode the textual data
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])

#set the dataset to be of similar datatype
values = values.astype('float32')

#normalise the inputs
from sklearn.preprocessing import MinMaxScaler
normaliser = MinMaxScaler(feature_range=(0, 1))
normalised_values = normaliser.fit_transform(values)


#now to set the input and output 
from pandas import DataFrame
normalised_values = DataFrame(normalised_values)

#shift the dataset down by one record and append the whole dataset to temp 
temp = list()
temp.append(normalised_values.shift(1)) #first column becomes Nan

#create column names for the dataset appended in temp and these dataset is
#to be considered as the dataset at (t-1) timestep
names = ['var{}(t-1)'.format(i) for i in range(8)]

#append again without shifting
temp.append(normalised_values)
#this dataset will be considered as dataset at t timestep
names += ['var{}(t)'.format(i) for i in range(8)]

#temp is now a list a two dataframes
#we append them column-wise into one dataset
from pandas import concat
agg_temp = concat(temp, axis=1)
agg_temp.columns = names
agg_temp.dropna(inplace=True)
#of the timestep t data, we only keep the first column and remove the rest
agg_temp.drop(agg_temp.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)

#we finally have the dataset we can work with
final_dataset = agg_temp.values

#split into train and test dataset
split = 365*24*2
train = final_dataset[:split, :]#first two years as train set
test =  final_dataset[split:, :]

#split the input and output
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

#reshape the dataset for RNN (samples,timestep,features)
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

print("\n\n\n\nDataset prepared------------------\n")


'''


implementing the lstm model


'''

#creating object of the Sequential class
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
model = Sequential()

#setting lstm as the model, with 60 neurons
model.add(LSTM(50,input_shape=(train_X.shape[1], train_X.shape[2])))

model.add(Dense(1)) #number of layers

model.compile(loss='mae', optimizer='adam')


print("RUNNING THE MODEL-------------------------")
#fit the dataset into the model
history = model.fit(train_X, train_y, epochs=50, batch_size=72, 
                    validation_data=(test_X, test_y), verbose=2, shuffle=False)


'''display the result'''
print("\n\n\n")
#plot the graph of train and test set
from matplotlib import pyplot
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

#get the output of the test data again to be used for rmse calcultion
y = model.predict(test_X)


#reshape the test set back to 2D from 3D
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
#append the rest of the data with the output
from numpy import concatenate
inv_y = concatenate((y, test_X[:, 1:]), axis=1)
#invert the the whole data; from normalised to original
#so that we get the expected original PM2.5 output values
inv_y = normaliser.inverse_transform(inv_y)
#take the first column
inv_y = inv_y[:,0]

#similarly invert scale the actual test data
test_y = test_y.reshape((len(test_y), 1))
inv_test_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_test_y = normaliser.inverse_transform(inv_test_y)
inv_test_y = inv_test_y[:,0]

# calculate RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(inv_test_y, inv_y))
print('Test RMSE: %.3f' % rmse)
