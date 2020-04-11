# ANN
import keras
 
#Housekeeping\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os

# Import data
raw_data_path = os.path.join(os.pardir, '..', 'data', 'raw') 
churnDF = pd.read_csv(os.path.join(raw_data_path, 'Churn_Modelling.csv'))

churnDF.drop(axis='columns', columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)

# ###  OneHotEncoder (using get_dummies)
churnDF = pd.get_dummies(churnDF, columns=['Geography', 'Gender'], prefix=["Geography", "Gender"])

#DOESN'T WORK (beacuse you CANNOT use OneHotEncoder to encode a string column)
# from sklearn.preprocessing import OneHotEncoder
# oneHotEncoder = OneHotEncoder(categorical_features=['Geography', 'Gender'])
# chrunDF2 = oneHotEncoder.fit_transform(churnDF)
# ValueError: could not convert string to float: 'France'

# ### Split
from sklearn.model_selection import train_test_split

X = churnDF.drop('Exited', axis='columns')
y = churnDF.Exited

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=0)

# ### StandardScaler (do this after splitting)
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
X_train = standardScaler.fit_transform(X_train)
X_test = standardScaler.fit_transform(X_test)

# ### ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

# #### BUILD
classifier = Sequential()

# ##### Add first hidden layer
classifier.add(Dense(output_dim=7, init='uniform', activation='relu', input_dim=13))   # 13+1/2 = 7 layers
# ##### Add second hidden layer
classifier.add(Dense(output_dim=7, init='uniform', activation='relu')) 
# ##### Add output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid')) # use submax for mulitple label classes
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# #### TRAIN/FIT
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100) #Epoch's are executed
 
# #### PREDICT
prediction = classifier.predict(X_test)

# ##### Convert prediction probability values to boolean
prediction = prediction > 0.5  # values greater than 0.5  is set to True, else, False

# #### EVALUATE
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, prediction)
ascore = accuracy_score(y_test, prediction)

# convert y_test values to boolean
y_test2 = y_test > 0
 
cm2 = confusion_matrix(y_test2, prediction)
ascore2 = accuracy_score(y_test2, prediction)

#single_observation_array = np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])   #create 2 dimensional array
single_observation_array = np.array([[700, 40, 3, 60000.00, 2, 1, 1, 50000, 1,0,0, 1,0]])   #create 2 dimensional array

single_observation_array_scaled = standardScaler.fit_transform(single_observation_array)
single_observation_array_scaled

single_prediction = classifier.predict(single_observation_array_scaled)
single_prediction > 0.5

# ### K-FOLD CROSS VALIDATION
#from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    tempClassifier = Sequential()
    tempClassifier.add(Dense(output_dim=7, init='uniform', activation='relu', input_dim=13))   # 13+1/2 = 7 layers
    tempClassifier.add(Dense(output_dim=7, init='uniform', activation='relu')) 
    tempClassifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid')) # use submax for mulitple label classes
    tempClassifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return tempClassifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size=5, nb_epoch=100)

#ERROR:
#accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, n_jobs=-1)
#BrokenProcessPool: A task has failed to un-serialize. Please ensure that the arguments of the function are all picklable.


accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train) #Epochs are executed 
accuracies.mean() 
accuracies.std()  # our variance is small;  overfitting DID NOT happen

# ### Dropout Regularization (is THE solutiorn for overfitting in deep learning)
 
from keras.layers import Dropout

# NOTES
# Overfitting happens when there is a large difference in accuracies of training set v/s test set

# Overfitting happens in K-fold cross validation, if the variance is high (this is because of difference in values of accuracies (some low values and some large values))

#With Dropout, some random neurons are disabled for each iteration
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier1():
    tempClassifier = Sequential()
    tempClassifier.add(Dense(output_dim=7, init='uniform', activation='relu', input_dim=13))   # 13+1/2 = 7 layers
    tempClassifier.add(Dropout(p=0.5))  #10% dropped
    tempClassifier.add(Dense(output_dim=7, init='uniform', activation='relu')) 
    tempClassifier.add(Dropout(p=0.3))  #10% dropped
    tempClassifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid')) # use submax for mulitple label classes
    tempClassifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return tempClassifier

classifier1 = KerasClassifier(build_fn=build_classifier1, batch_size=5, nb_epoch=100)
accuracies1 = cross_val_score(estimator=classifier, X=X_train, y=y_train)
accuracies1.std()


# ### PARAMETER TUNING (GRID SEARCH)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
def build_classifier_for_grid_search(p, optimizer):
    tempClassifier = Sequential()
    tempClassifier.add(Dense(output_dim=7, init='uniform', activation='relu', input_dim=13))   # 13+1/2 = 7 layers
    tempClassifier.add(Dropout(p=p))  #10% dropped
    tempClassifier.add(Dense(output_dim=7, init='uniform', activation='relu')) 
    tempClassifier.add(Dropout(p=p))  #10% dropped
    tempClassifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid')) # use submax for mulitple label classes
    tempClassifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return tempClassifier


gridSearchClassifier = KerasClassifier(build_fn=build_classifier_for_grid_search)

parameters = { 'batch_size': [25, 32],
                'nb_epoch': [100, 200],
                'optimizer':['adam', 'rmsprop'],
                'p': [0.1, 0.2]}
 
gridSearch = GridSearchCV(estimator=gridSearchClassifier, param_grid=parameters, scoring='accuracy', cv=10)
gridSearchModel = gridSearch.fit(X_train, y_train)

bestParameter = gridSearchModel.best_params_
bestAccuracy = gridSearchModel.best_score_


#END

 
get_ipython().system('jupyter notebook nbconvert --to script UDEMY_DEEP_LEARNING_A-Z_ANN_KIRILL_2.ipynb')

