# # HOUSEKEEPING
import os
import pandas as pd
import numpy as np



def readFiles():
    # set raw data path
    raw_data_path = os.path.join(os.path.pardir, 'data', 'raw') #'../data/raw
    # read train.csv & build trainDF
    train_data_path = os.path.join(raw_data_path, 'train.csv') #'../data/raw/train.csv'
    trainDF = pd.read_csv(train_data_path, index_col='PassengerId')
    # read test.csv & build testDF
    test_data_path = os.path.join(raw_data_path, 'test.csv') #'../data/raw/test.csv'
    testDF = pd.read_csv(test_data_path, index_col='PassengerId')
    # set Survived = -888 for test dataset
    testDF['Survived'] = -888
    # combine trainDF & testDF so that you can process them together
    combinedDF = pd.concat((trainDF, testDF), sort=True)  #axis=0 (default)
    print(combinedDF.head())
    return combinedDF

def printDF(myDF):
    print ('in printDF(myDF)')
    print('Columns:' + myDF.columns)
    print(myDF.head())

title_dictionary = {'mr'	:	'mr',
 'mrs'	:	 'mrs',
 'miss'	:	 'miss',
 'master'	:	 'master',
 'don'	:	 'sir',
 'rev'	:	 'sir',
 'dr'	:	 'officer',
 'mme'	:	 'mrs',
 'ms'	:	 'mrs',
 'major'	:	 'officer',
 'lady'	:	 'lady',
 'sir'	:	 'sir',
 'mlle'	:	 'miss',
 'col'	:	 'officer',
 'capt'	:	 'officer',
 'the countess'	:	 'lady',
 'jonkheer'	:	 'sir',
 'dona'	:	 'lady'}

# Routines:
def getTitle(name):
    fname = name.split(',')[1]
    #return fname       # ' Mr. Owen Harris'
    title = fname.split('.')[0]
    #return title.strip().lower()
    return title_dictionary.get(title.strip().lower())


# def getStandardTitle(title):
#     return title_dictionary.get(title)


def getDeck(cabin):
    return np.where(pd.notnull(cabin), str(cabin)[0].upper(), 'Z')
# Routines:


def writeFiles(combinedDF):
    # set processed data path
    processed_data_path = os.path.join(os.path.pardir, 'data', 'processed') #'../data/processed
    # filter out train data and save it
    write_train_data_path = os.path.join(processed_data_path, 'train.csv') #'../data/processed/train.csv'
    combinedDF.loc[combinedDF.Survived != -888].to_csv(write_train_data_path)
    # filter out test data and save it
    write_test_data_path = os.path.join(processed_data_path, 'test.csv') #'../data/processed/test.csv'
    combinedDF.loc[combinedDF.Survived == -888].to_csv(write_test_data_path)

def fillMissingValues(combinedDF):
    #Embarked
    combinedDF.Embarked.fillna('C', inplace=True)
    #Fare
    median_fare = combinedDF[('Embarked' == 'S') & (combinedDF['Pclass'] == 3)].Fare.median()
    combinedDF.Fare.fillna(median_fare, inplace=True)
    #Age
    combinedDF.Age.fillna(combinedDF.groupby('Title').Age.transform('median'), inplace=True)
    return combinedDF


def processData(combinedDF):      #combinedDF will be passed along
    ### Method Chaining:
    print('In processData...')
    print(combinedDF.columns)
    # Creating 'Title' had to be done earlier beacuse 'IsMother' uses Title

    combinedDF = combinedDF.assign(Title = lambda x: x.Name.map(getTitle)) # create 'Title' feature
    combinedDF = combinedDF.pipe(fillMissingValues) \
                    .assign(Adult = lambda x: np.where(x.Age >= 18, 'Yes', 'No')) \
                    .assign(FamilySize =  combinedDF.Parch + combinedDF.SibSp  + 1) \
                    .assign(IsMother = np.where(((combinedDF.Sex == 'female') &
                                      (combinedDF.Title != 'miss') &
                                      (combinedDF.Age > 18) &
                                      (combinedDF.Parch > 0)), 1,0)) \
                    .assign(Cabin = lambda x: np.where(x.Cabin == 'T', np.nan, x.Cabin)) \
                    .assign(Deck = lambda x: x.Cabin.map(getDeck)) \
                    .assign(IsMale = lambda x: np.where(x.Sex == 'male', 1, 0))
    combinedDF = combinedDF.pipe(pd.get_dummies, columns=['Deck', 'Pclass', 'Title', 'Embarked', 'Adult']) \
                           .drop(['Cabin','Name','Ticket','Parch','SibSp','Sex'], axis=1) \
                           .pipe(sortColumns)

    printDF(combinedDF)
    print('In processData DONE')
    return combinedDF

def sortColumns(combinedDF):
    # Reorder columns
    t_columns = combinedDF.columns.tolist()
    t_columns.remove('Survived')
    t_columns.sort()
    #t_columns.append('Survived') # this adds Survived at the end
    t_columns = ['Survived'] + t_columns
    combinedDF = combinedDF[t_columns]
    return combinedDF


if __name__ == '__main__':
    print('in __main__')

    myDF = readFiles()
    myDF = processData(myDF)
    # writeFiles(myDF)  # enable this line overwrite the files
    print('in __main__  DONE')
