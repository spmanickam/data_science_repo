
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#HOUSEKEEPING" data-toc-modified-id="HOUSEKEEPING-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>HOUSEKEEPING</a></span></li><li><span><a href="#DATASCIENCE-PROJECT-CYCLE" data-toc-modified-id="DATASCIENCE-PROJECT-CYCLE-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>DATASCIENCE PROJECT CYCLE</a></span></li><li><span><a href="#EXPLORING-AND-PROCESSING-DATA--PART1" data-toc-modified-id="EXPLORING-AND-PROCESSING-DATA--PART1-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>EXPLORING AND PROCESSING DATA- PART1</a></span><ul class="toc-item"><li><span><a href="#EXPLORATORY-DATA-ANALYSIS-(EDA)" data-toc-modified-id="EXPLORATORY-DATA-ANALYSIS-(EDA)-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>EXPLORATORY DATA ANALYSIS (EDA)</a></span><ul class="toc-item"><li><span><a href="#Combine-Train-&amp;-Test" data-toc-modified-id="Combine-Train-&amp;-Test-3.1.1"><span class="toc-item-num">3.1.1&nbsp;&nbsp;</span>Combine Train &amp; Test</a></span></li><li><span><a href="#Basic-Structure" data-toc-modified-id="Basic-Structure-3.1.2"><span class="toc-item-num">3.1.2&nbsp;&nbsp;</span>Basic Structure</a></span></li><li><span><a href="#Summary-Statistics" data-toc-modified-id="Summary-Statistics-3.1.3"><span class="toc-item-num">3.1.3&nbsp;&nbsp;</span>Summary Statistics</a></span><ul class="toc-item"><li><span><a href="#Numerical-Features" data-toc-modified-id="Numerical-Features-3.1.3.1"><span class="toc-item-num">3.1.3.1&nbsp;&nbsp;</span>Numerical Features</a></span></li><li><span><a href="#Categorical-Features" data-toc-modified-id="Categorical-Features-3.1.3.2"><span class="toc-item-num">3.1.3.2&nbsp;&nbsp;</span>Categorical Features</a></span></li></ul></li></ul></li></ul></li><li><span><a href="#EXPLORING-AND-PROCESSING-DATA--PART2" data-toc-modified-id="EXPLORING-AND-PROCESSING-DATA--PART2-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>EXPLORING AND PROCESSING DATA- PART2</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#Distributions" data-toc-modified-id="Distributions-4.0.1"><span class="toc-item-num">4.0.1&nbsp;&nbsp;</span>Distributions</a></span><ul class="toc-item"><li><span><a href="#Univariate:-Histogram-&amp;-KDE-(probability-based)" data-toc-modified-id="Univariate:-Histogram-&amp;-KDE-(probability-based)-4.0.1.1"><span class="toc-item-num">4.0.1.1&nbsp;&nbsp;</span>Univariate: Histogram &amp; KDE (probability based)</a></span></li><li><span><a href="#Bivariate:-Scatter-plots" data-toc-modified-id="Bivariate:-Scatter-plots-4.0.1.2"><span class="toc-item-num">4.0.1.2&nbsp;&nbsp;</span>Bivariate: Scatter plots</a></span></li></ul></li><li><span><a href="#Grouping(Aggregarion)" data-toc-modified-id="Grouping(Aggregarion)-4.0.2"><span class="toc-item-num">4.0.2&nbsp;&nbsp;</span>Grouping(Aggregarion)</a></span></li><li><span><a href="#Crosstab-(for-categorical-variables)" data-toc-modified-id="Crosstab-(for-categorical-variables)-4.0.3"><span class="toc-item-num">4.0.3&nbsp;&nbsp;</span>Crosstab (for categorical variables)</a></span></li><li><span><a href="#Pivot-Table" data-toc-modified-id="Pivot-Table-4.0.4"><span class="toc-item-num">4.0.4&nbsp;&nbsp;</span>Pivot Table</a></span></li></ul></li></ul></li><li><span><a href="#EXPLORING-AND-PROCESSING-DATA--PART3" data-toc-modified-id="EXPLORING-AND-PROCESSING-DATA--PART3-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>EXPLORING AND PROCESSING DATA- PART3</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#DATA-MUNGING-(missing-values,-outliers)" data-toc-modified-id="DATA-MUNGING-(missing-values,-outliers)-5.0.1"><span class="toc-item-num">5.0.1&nbsp;&nbsp;</span>DATA MUNGING (missing values, outliers)</a></span><ul class="toc-item"><li><span><a href="#Missing-values" data-toc-modified-id="Missing-values-5.0.1.1"><span class="toc-item-num">5.0.1.1&nbsp;&nbsp;</span>Missing values</a></span></li><li><span><a href="#DROPPING-A-COLUMN-AND-ADDING-IT-BACK" data-toc-modified-id="DROPPING-A-COLUMN-AND-ADDING-IT-BACK-5.0.1.2"><span class="toc-item-num">5.0.1.2&nbsp;&nbsp;</span>DROPPING A COLUMN AND ADDING IT BACK</a></span></li><li><span><a href="#Impute-missing-values" data-toc-modified-id="Impute-missing-values-5.0.1.3"><span class="toc-item-num">5.0.1.3&nbsp;&nbsp;</span>Impute missing values</a></span></li></ul></li><li><span><a href="#WORKING-WITH-OUTLIERS" data-toc-modified-id="WORKING-WITH-OUTLIERS-5.0.2"><span class="toc-item-num">5.0.2&nbsp;&nbsp;</span>WORKING WITH OUTLIERS</a></span><ul class="toc-item"><li><span><a href="#Outlier-Detection" data-toc-modified-id="Outlier-Detection-5.0.2.1"><span class="toc-item-num">5.0.2.1&nbsp;&nbsp;</span>Outlier Detection</a></span></li><li><span><a href="#Outlier-Treatment" data-toc-modified-id="Outlier-Treatment-5.0.2.2"><span class="toc-item-num">5.0.2.2&nbsp;&nbsp;</span>Outlier Treatment</a></span></li><li><span><a href="#Binning" data-toc-modified-id="Binning-5.0.2.3"><span class="toc-item-num">5.0.2.3&nbsp;&nbsp;</span>Binning</a></span></li></ul></li><li><span><a href="#FEATURE-ENGINEERING" data-toc-modified-id="FEATURE-ENGINEERING-5.0.3"><span class="toc-item-num">5.0.3&nbsp;&nbsp;</span>FEATURE ENGINEERING</a></span><ul class="toc-item"><li><span><a href="#FEATURE-ENCODING-(for-categorical-features)" data-toc-modified-id="FEATURE-ENCODING-(for-categorical-features)-5.0.3.1"><span class="toc-item-num">5.0.3.1&nbsp;&nbsp;</span>FEATURE ENCODING (for categorical features)</a></span></li><li><span><a href="#Drop-and-Reorder-column" data-toc-modified-id="Drop-and-Reorder-column-5.0.3.2"><span class="toc-item-num">5.0.3.2&nbsp;&nbsp;</span>Drop and Reorder column</a></span></li><li><span><a href="#SAVE-DATAFRAME-TO-FILE" data-toc-modified-id="SAVE-DATAFRAME-TO-FILE-5.0.3.3"><span class="toc-item-num">5.0.3.3&nbsp;&nbsp;</span>SAVE DATAFRAME TO FILE</a></span></li></ul></li><li><span><a href="#PLOTTING" data-toc-modified-id="PLOTTING-5.0.4"><span class="toc-item-num">5.0.4&nbsp;&nbsp;</span>PLOTTING</a></span></li><li><span><a href="#SUBPLOTS" data-toc-modified-id="SUBPLOTS-5.0.5"><span class="toc-item-num">5.0.5&nbsp;&nbsp;</span>SUBPLOTS</a></span></li></ul></li></ul></li></ul></div>

# # HOUSEKEEPING

# In[3]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from IPython.display import Image
from IPython.core.display import HTML  


# # DATASCIENCE PROJECT CYCLE

# In[2]:


Image(filename='images/data_science_project_lifecycle2.png', height=400, width=600)


# # EXPLORING AND PROCESSING DATA- PART1
# OVERVIEW OF STEPS:
    1) Import data
    2) Exploratory Data Analysis: 
         --> explore the data using some basic statistics & basic visualizations
         --> Identify Missing values
         --> Identify Outliers
    3) Data munging: 
        --> Take care of the issues unconvered during exploratory data analysis
        --> Look for potential issues and fix/resolve them
    4) Feature Engineering
        --> Process existing features
        --> Generate new features
    5) Advanced Visualizations
        --> To gain more insights into the data
        --> May result in doing more feature engineering
    6) Perform Data Science
# ## EXPLORATORY DATA ANALYSIS (EDA)

# In[3]:


Image(filename='images/eda_chart.png', height=400, width=600)


# In[4]:


Image(filename='images/eda_chart2.png', height=400, width=600)


# In[ ]:


EDA -> Basic Structure (observations(rows), features(columns), data types, head/tail )
-> Summary statistics
    --> Numerical
        ---> Centrality measure(mean, median(middle value))
        ---> Dispersion measure(range(max-min), percentiles, variance, SD)
    --> Categorical
        ---> Total count
        ---> Unique count
        ---> Category counts and proportions
        ---> Per category statistics(grouping)
-> Distributions
    --> Univariate
        ---> Historgram
        ---> Boxplot
        ---> Kernel Density Estimates(KDE) plot
    --> Bivariate
        ---> Scatter plot
    --> Multivariate
        ---> ???
-> Grouping
-> Crosstabs
--> Pivot table


# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


#WORKS
# !pwd
# OUTPUT: /Users/pinky/Downloads/LEARNING/PLURALSIGHT_Python_data_science_Abhishek_kumar/module2/titanic/notebooks
# !ls /Users/pinky/Downloads/LEARNING/PLURALSIGHT_Python_data_science_Abhishek_kumar/module2/titanic/data/raw
# OUTPUT: gender_submission.csv test.csv              train.csv
    
# path ='/Users/pinky/Downloads/LEARNING/PLURALSIGHT_Python_data_science_Abhishek_kumar/module2/titanic/data/raw'
# train_data = pd.read_csv(path + '/train.csv')


# In[4]:


raw_data_path = os.path.join(os.path.pardir, 'data', 'raw') #'../data/raw

train_data_path = os.path.join(raw_data_path, 'train.csv') #'../data/raw/train.csv'
trainDF = pd.read_csv(train_data_path, index_col='PassengerId')

test_data_path = os.path.join(raw_data_path, 'test.csv') #'../data/raw/test.csv'
testDF = pd.read_csv(test_data_path, index_col='PassengerId')


# ### Combine Train & Test 

# In[48]:


trainDF.info()


# In[49]:


testDF.info()


# In[5]:


# Join Train & Test datasets
testDF['Survived'] = -888


# In[51]:


testDF.info()


# In[6]:


combinedDF = pd.concat((trainDF, testDF), sort=True)  #acis=0 (default)


# In[53]:


trainDF.count().tolist(), testDF.count()


# In[54]:


trainDF.keys()


# In[55]:


trainDF.count().tolist()


# In[56]:


testDF.count().tolist()


# In[57]:


combinedDF.head()

Titanic Data Dictionary
Variable	Definition	
survival	Survival (0 = No, 1 = Yes)
pclass	    Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
sex	        Sex	
Age	        Age in years	
sibsp	    # of siblings / spouses aboard the Titanic	
parch	    # of parents / children aboard the Titanic	
ticket	    Ticket number	
fare	    Passenger fare	
cabin	    Cabin number	
embarked	Port of Embarkation	(C = Cherbourg, Q = Queenstown, S = Southampton)
# ### Basic Structure

# In[74]:


type(combinedDF.Name)


# In[75]:


combinedDF.index


# In[76]:


combinedDF.columns


# In[77]:


combinedDF.Name[0:5]  #select first 5 from series


# In[78]:


df2.head()


# In[79]:


combinedDF[['Name', 'Age']][6:8]  # select multiple columns


# In[80]:


trainDF.loc[800:804] # select based on index values(rows)


# In[81]:


trainDF.iloc[800:804]  #select based on index values(rows) & columns


# In[85]:


trainDF.loc[2:5, ['Age', 'Pclass']]  #select based on index values(rows) & columns


# In[83]:


trainDF.iloc[2:5,3:8] # use iloc for postion based indexing


# In[86]:


# filter
male_passengers = combinedDF.loc[combinedDF.Sex == 'male']
len(male_passengers)


# In[87]:


male_passengers.head()


# In[88]:


# filter
first_class_male_passengers = combinedDF.loc[(combinedDF.Sex == 'male') & (combinedDF.Pclass == 1)]
len(first_class_male_passengers)


# ### Summary Statistics

# In[89]:


combinedDF.describe()


# #### Numerical Features

# ##### Centrality measures

# In[90]:


combinedDF.Age.mean()


# In[91]:


combinedDF.Age.median()


# ##### Dispersion measures

# In[92]:


combinedDF.Fare.min(), combinedDF.Fare.max()


# In[93]:


combinedDF.Fare.max() - combinedDF.Fare.min()  # range


# In[94]:


combinedDF.Fare.quantile(0.25), combinedDF.Fare.quantile(0.5), combinedDF.Fare.quantile(0.75)


# In[95]:


combinedDF.Fare.var(), combinedDF.Fare.std()  #variance & SD


# In[96]:


Image(filename='images/box_whisker_plot.png', height=400, width=600)


# In[97]:


get_ipython().run_line_magic('matplotlib', 'inline')
combinedDF.Fare.plot(kind='box')


# #### Categorical Features

# In[98]:


# Counts
combinedDF.describe(include='all')


# In[99]:


combinedDF.Sex.value_counts()


# In[100]:


combinedDF.Sex.value_counts(normalize=True) # proportions


# In[101]:


combinedDF.Survived.value_counts()


# In[102]:


combinedDF[combinedDF.Survived != -888].Survived.value_counts()


# In[103]:


combinedDF.Pclass.value_counts()


# In[104]:


combinedDF.Pclass.value_counts().plot(kind='bar')


# In[105]:


combinedDF.Pclass.value_counts().plot(kind='bar', rot = 0, title='Passenger count per class');


# In[322]:


combinedDF.Embarked.describe()


# # EXPLORING AND PROCESSING DATA- PART2

# ### Distributions

# #### Univariate: Histogram & KDE (probability based)

# In[106]:


Image(filename='images/univariate_distributions.png', height=400, width=600)


# In[107]:


Image(filename='images/skewness.png', height=400, width=600)

Note: Pandas ignores missing values
# In[125]:


combinedDF.Age.plot(kind='hist', bins=20, title='Age Histogram', figsize=(4,2));


# In[124]:


combinedDF.Age.plot(kind='kde',title='Age KDE', figsize=(4,2));


# In[123]:


combinedDF.Fare.plot(kind='hist', bins=20, title='Fare Histogram', figsize=(4,2));


# In[122]:


combinedDF.Fare.plot(kind='kde',title='Fare KDE', figsize=(4,2));


# In[116]:


# Compute skewness
combinedDF.Age.skew(), combinedDF.Fare.skew()  # Fare is right sckewed (verymuch > 0) 


# #### Bivariate: Scatter plots

# In[120]:


combinedDF.plot.scatter(x='Age', y='Fare', alpha=0.1, figsize=(4,2));


# In[126]:


combinedDF.plot.scatter(x='Age', y='Fare', alpha=0.1, figsize=(4,2));


# In[ ]:


#NOTES: No correlation between Dare & Age


# In[128]:


combinedDF.plot.scatter(x='Pclass', y='Fare', alpha=0.3, figsize=(4,2));


# In[ ]:


#NOTES: Only 3 Pclasses; 
#       In 1st class, some people have got cheap tickets  
#       Passengers in 2nd & 3rd class paid fares between 0 and 100


# ### Grouping(Aggregarion)

# In[140]:


combinedDF.groupby('Sex').mean()  #display app columns


# In[141]:


combinedDF.groupby('Sex').Age.mean()


# In[142]:


combinedDF.groupby(['Pclass','Sex'])['Fare', 'Age'].median() # display selected columns


# In[143]:


combinedDF.groupby(['Pclass']).agg({'Fare':'mean', 'Age':'median'})  # using agg


# In[148]:


aggregations = {
    'Fare': {
            'mean_fare' : 'mean' ,
            'max_fare' : max,
            'min_fare' : np.min
            },
    'Age': {
        'range_age' : lambda x: max(x) - min(x)
    }
}


# In[151]:


combinedDF.groupby('Pclass').agg(aggregations)


# ### Crosstab (for categorical variables)

# In[153]:


pd.crosstab(combinedDF.Sex, combinedDF.Pclass)


# In[155]:


pd.crosstab(combinedDF.Sex, combinedDF.Pclass).plot(kind='bar')


# ### Pivot Table

# In[160]:


Image(filename='images/pivot_table.png', height=300, width=500)


# In[167]:


# index --> 1
# cokumns --> 2
# values --> 3
# addfunc --> 4

combinedDF.pivot_table(index='Sex', columns='Pclass', values='Age', aggfunc='mean')


# In[164]:


# Same result as pivot table
combinedDF.groupby(['Sex', 'Pclass']).Age.mean()


# In[166]:


# Same result as pivot table
combinedDF.groupby(['Sex', 'Pclass']).Age.mean().unstack()


# # EXPLORING AND PROCESSING DATA- PART3

# ### DATA MUNGING (missing values, outliers)

# In[170]:


Image(filename='images/munging.png', height=300, width=500)


# #### Missing values

# In[ ]:


Issue: inaccurate analysis & most models won't work with missing values
Solution: 
      -> Deletion 
      -> Imputation
         --> Mean imputation  (average) (result IS impacted by outliers)
         --> Median imputation (middle value) (result NOT impacted by outliers)
         --> Mode imputation (most occurring) (for categorical features)
         --> Forward fill (replace with previous value)
         --> Backward fill (replace with next value)
         --> Predictive Model (predict using models)


# In[172]:


combinedDF.info()  # Age, CAbin, Embarked, Fare have missing values (< 1309)


# In[202]:


len(combinedDF.Embarked.unique())


# In[203]:


combinedDF.Embarked.unique()


# In[218]:


combinedDF.Embarked.value_counts(dropna=False, sort=False)  # BEST RESULT for count unique values including nan's


# In[212]:


combinedDF.Embarked.value_counts().to_frame()


# In[207]:


combinedDF.groupby('Embarked').size() 


# In[214]:


combinedDF.Embarked.nunique(dropna=False)


# In[215]:


combinedDF[combinedDF.Embarked.isnull()] # BEST RESULT for displaying obervations with nan's


# In[219]:


#NOTE: Both the passenger wiht Embarked=NaN survived


# In[233]:


# which embarkement point has most survivers?
pd.crosstab(trainDF.Embarked, trainDF.Survived)


# In[232]:


combinedDF.groupby(['Survived', 'Embarked']).Embarked.count().to_frame().T


# #### DROPPING A COLUMN AND ADDING IT BACK

# In[291]:


# WORKS:
# DROPPING A COLUMN AND ADDING IT BACK
# df1 = df2 = df3 = combinedDF.copy()
#df1.Embarked.fillna('X' , inplace = False) # no change to df1; hence waste
#df2.Embarked.fillna('X' , inplace = True) # both NaNs for Embarked is replaced with 'X'; use this always
 
#series1 = df3.Embarked.fillna('X' , inplace = False) # new Series object gets created # same as df3.Embarked  
#df3.drop('Embarked', inplace=True, axis='columns')  # drop column 'Embarked'
#df3['Embarked1'] = series1  # add back the column
#NOTE: series1.index === df3.index


# #### Impute missing values 

# In[ ]:


# NOTE: Use median instead of mean to avoid imppact of extreme values(outliers)


# In[ ]:


# NOTE: inplace = True will change existing dataframe
#        inplace = False --> pandas will create a copy of existing DF 


# ##### IMPUTE MISSING VALUE FOR Embarked

# In[309]:


#OPTION1 : Replace with mode, as Embarked is a categorical variable
combinedDF.Embarked.mode()


# In[7]:


# Impute missing values 
# combinedDF.loc[combinedDF.Embarked.isnull, 'Embarked'] = 'S'
combinedDF.Embarked.fillna('S', inplace=True)


# In[314]:


#OPTION2
# Find out the Fare for each class for each embarkement point; we know the fare paid by passengers 
# with Embarked = NaN is 80; so select the Embarked value class = 1 and with median closest to 80; 
# in this case, it is C (76) 
combinedDF.groupby(['Pclass', 'Embarked']).Fare.median()


# In[321]:


combinedDF.pivot_table(index='Pclass', columns='Embarked', values='Fare', aggfunc='median')


# In[8]:


combinedDF.Embarked.fillna('C', inplace=True)


# In[331]:


combinedDF[combinedDF.Embarked.isnull()]


# In[332]:


combinedDF.info()


# ##### IMPUTE MISSING VALUE FOR Fare

# In[334]:


combinedDF[combinedDF.Fare.isnull()]


# In[335]:


combinedDF.pivot_table(index='Pclass', columns='Embarked', values='Fare', aggfunc='median')


# In[336]:


combinedDF.pivot_table(index='Pclass', columns='Embarked', values='Fare', aggfunc='mean')


# In[351]:


combinedDF[(combinedDF['Embarked'] == 'S') & (combinedDF['Pclass'] == 3)].Fare.value_counts(ascending=False).head()


# In[355]:


combinedDF.loc[(combinedDF.Embarked =='S') & (combinedDF.Pclass == 3), 'Fare'].median()


# In[358]:


combinedDF.Fare.fillna(8.05, inplace=True)


# In[359]:


combinedDF.info()


# ##### IMPUTE MISSING VALUE FOR Age

# In[363]:


# Count of NaNs for Age column
combinedDF.Age.value_counts(dropna=False, ascending=False).head()


# In[367]:


combinedDF[combinedDF.Age.isnull()].head()


# In[370]:


combinedDF[combinedDF.Age.isna()].head()


# In[380]:


len(combinedDF[combinedDF.Age.isna()])


# In[382]:


combinedDF.Age.plot(kind='hist');


# In[383]:


combinedDF.Age.mean()


# ##### OPTION1 : impute with mean

# In[ ]:


# OPTION1: impute with mean is not a good option as there are outliers
#combinedDF.Age.fillna(combinedDF.Age.mean, inplace=True)


# ##### OPTION2: impute with median age per gender

# In[385]:


combinedDF.groupby('Sex').Age.median()


# In[398]:


combinedDF.Age.plot.box();


# In[400]:


combinedDF.boxplot('Age', 'Sex');


# In[402]:


combinedDF[combinedDF.Age.notnull()].boxplot('Age', 'Sex');


# In[399]:


# OPTION2: impute with median per sex is not a good option as there no much difference between medians (287 & 28)
# DOESN'T WORK:combinedDF.Age.fillna(combinedDF.groupby('Sex').Age.median(), inplace=True)
# WORKS: combinedDF.Age.fillna(combinedDF.groupby('Sex').Age.transform('median'), inplace=True)


# In[403]:


combinedDF.groupby('Sex').Age.median()


# In[405]:


combinedDF.groupby('Sex').Age.transform('median').head()


# In[406]:


df5 = combinedDF.copy()


# In[408]:


df5.Age.value_counts(dropna=False).head()


# In[409]:


df5.Age.fillna(combinedDF.groupby('Sex').Age.median(), inplace=True)


# In[410]:


df5.Age.value_counts(dropna=False).head()


# In[411]:


df5.Age.fillna(combinedDF.groupby('Sex').Age.transform('median'), inplace=True)


# In[412]:


df5.Age.value_counts(dropna=False).head()


# ##### OPTION3: impute with median per Pclass 

# In[420]:


combinedDF.groupby(['Pclass']).Age.median()


# In[421]:


combinedDF.boxplot('Age', 'Pclass')


# In[423]:


combinedDF[combinedDF.Age.notnull()].boxplot('Age', 'Pclass')


# In[ ]:


# OPTION3: impute with median Age per Pclass is not a bad choice 
# ombinedDF.Age.fillna(combinedDF.groupby('Pclass').Age.transform('median'), inplace=True)


# ##### OPTION4: impute with median age based on title(which is part of the name)

# In[427]:


combinedDF.Name.head()


# In[501]:


def getTitle(name): 
    fname = name.split(',')[1] 
    #return fname       # ' Mr. Owen Harris'
    title = fname.split('.')[0]
    return title.strip().lower()


# In[502]:


getTitle('Braund, Mr. Owen Harris')


# In[467]:


combinedDF.Name.map(lambda x: getTitle(x)).head()


# In[474]:


combinedDF.Name.map(lambda x: getTitle(x)).unique()


# In[477]:


combinedDF.Name.map(lambda x: getTitle(x)).value_counts(ascending=False).head()


# In[479]:


combinedDF.Name.map(lambda x: getTitle(x)).unique().tolist()


# In[485]:


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


# In[487]:


title_dictionary.keys()


# In[488]:


title_dictionary.values()


# In[492]:


title_dictionary.get('dona')


# In[493]:


combinedDF.columns


# In[495]:


def getStandardTitle(title):
    return title_dictionary.get(title)


# In[496]:


getStandardTitle('dona')


# In[504]:


combinedDF['Title'] = combinedDF.Name.map(lambda x: getStandardTitle(getTitle(x)))


# In[505]:


combinedDF.head()


# In[508]:


combinedDF.boxplot('Age', 'Title')


# In[510]:


combinedDF.Age.value_counts(dropna=False).head()


# In[512]:


#OPTION4: impute with median age based on title(which is part of the name) is very good choice
combinedDF.Age.fillna(combinedDF.groupby('Title').Age.transform('median'), inplace=True)


# In[513]:


combinedDF.Age.value_counts(dropna=False).head()


# In[515]:


combinedDF.info()


# In[ ]:


# NOTE: Will ignore Cabin feature; hence no work needs to be done


# ### WORKING WITH OUTLIERS
How: Data entry, data processing, natural
Issue: Baised Analysis, Baised models
# #### Outlier Detection
Univariate:
 -> Histogram
 -> Boxplot
Bivariate
 -> Scatter plot
# #### Outlier Treatment
Removal
Transformation (log, sqrt)
Bins
Imputation
# ##### Age

# In[521]:


combinedDF.Age.plot(kind='hist', bins=20, figsize=(4,3));

NOTE:Age histogram shows that most values are in the range of 20-40. We can aslo see that there are few outliers with Age > 70. Checkout those recored:
# In[535]:


combinedDF.loc[combinedDF.Age>70]


# ##### Fare

# In[536]:


combinedDF.Fare.plot(kind='hist', bins=20)

NOTE: Some passengers have paid high price (500). Explore those
# In[540]:


combinedDF.Fare.plot(kind='box')


# In[547]:


combinedDF.loc[combinedDF.Fare == combinedDF.Fare.max()]

NOTE: All of these passengers paid max amount; also there have same Pclass & ticket number; and first 3 survived
      Eventhough these observations are outliers, they give us some interesting informationNOTE: Since Fare has very skewed distribution, we can apply some transformation to make it less skewed.
      Transformation is a very common way to treat outliers(at least to some extent)
      Since passenger fare can never by negative, we can try log transformation
# In[548]:


logFare = np.log(combinedDF.Fare + 1.0) # 1.0 is added to accommodate 0 fares (log(0) is undefined


# In[550]:


logFare.plot(kind='hist' ,bins=20)

NOTE: Above hist is less skewed
# #### Binning

# In[555]:


#TO DO: # NOT WORKINGcombinedDF.qcut(combinedDF.Fare, 4)


# ### FEATURE ENGINEERING
Process of transforming raw data to better representative features Types:
    -> Feature Transformation
    -> Feature Creation (using domain expertise)
    -> Feature Selection
# ##### New Feature: Adult

# In[563]:


combinedDF['Adult'] = np.where(combinedDF.Age >= 18, 'Yes', 'No')


# In[564]:


combinedDF.Adult.value_counts()


# In[567]:


pd.crosstab(combinedDF[combinedDF.Survived != -888].Survived, combinedDF[combinedDF.Survived != -888].Adult)

NOTE: 63 childresn survived & 54 children died
# ##### New Feature: Family size

# In[569]:


combinedDF['FamilySize'] =  combinedDF.Parch + combinedDF.SibSp  + 1 # 1 for self


# In[570]:


combinedDF['FamilySize'].plot(kind='hist')


# In[572]:


pd.crosstab(combinedDF[combinedDF.Survived != -888].Survived, combinedDF[combinedDF.Survived != -888].FamilySize)


# In[574]:


combinedDF.FamilySize.plot(kind='hist')


# In[575]:


combinedDF.loc[combinedDF.FamilySize >10]

NOTE: All observations have family size =11; first 7 didnt survive; hence most likely last 4 from test data woundn't have survuived
# ##### New feature: IsMother

# In[579]:


combinedDF['IsMother'] = np.where(((combinedDF.Sex == 'female') & 
                                  (combinedDF.Title != 'miss') & 
                                  (combinedDF.Age > 18) &
                                  (combinedDF.Parch > 0)), 1,0)


# In[581]:


combinedDF.IsMother.value_counts()


# In[583]:


pd.crosstab(combinedDF[combinedDF.Survived != -888].Survived, combinedDF[combinedDF.Survived != -888].IsMother)

NOTE: 39 mothers survived
# ##### New Feature: Deck

# In[590]:


combinedDF.Cabin.value_counts().head()


# In[586]:


combinedDF.Cabin.unique()

NOTE: You see cabin called 'T' and cabin with NaN
      Replace both of these with 'Z' # dont want to create 1 deck for 1 passenger
# In[591]:


combinedDF.loc[combinedDF.Cabin =='T']


# In[593]:


combinedDF.loc[combinedDF.Cabin =='T', 'Cabin'] = np.NaN


# In[594]:


combinedDF.loc[combinedDF.Cabin =='T']


# In[606]:


def getDeck(cabin):
    return np.where(pd.notnull(cabin), str(cabin)[0].upper(), 'Z')


# In[607]:


combinedDF['Deck'] = combinedDF.Cabin.map(lambda x: getDeck(x))


# In[610]:


combinedDF.Deck.value_counts()


# In[611]:


pd.crosstab(combinedDF[combinedDF.Survived != -888].Survived, combinedDF[combinedDF.Survived != -888].Deck)

NOTE: Survival rate is higher for decks B, C & E. So Deck is a good indicator
# In[612]:


combinedDF.info()

NOTE: columns with types object & category are categorical features
# #### FEATURE ENCODING (for categorical features)
-> converts categorical feature to numerical feature
# In[ ]:


Feature encoding methods  
    -> Binary Encoding (only 2 possible values for the feature)
       --> You can use np.where or One-Hot encoding 
    -> Label Encoding (multi category feature)
        --> Use integers to encode each level 
        --> Ex: Fare can be low(=1) , medium(=2), high(=3) 
        --> ML algoritm will consider High > Low beacuse 3 > 1
        --> Use label encoding for ordered categories(when you feature has some 
            kind of ascending/descending order(Ex: Fare))
    -> One-Hot Encoding
        --> Add 'n' features(columns) for each categories of your existing feature
        --> Ex: Embarkment(S, C, Q): Create new 3 features -> Is_S, Is_C, Is_Q 
        --> Use pd.get_dummies (After encoding, the original column is dropped automatically)


# In[613]:


combinedDF.Embarked.unique()


# ##### New Feature: IsMale (binary feature)
#     

# In[615]:


combinedDF['IsMale'] = np.where(combinedDF.Sex == 'male', 1,0)


# In[616]:


combinedDF.columns


# ##### New feature: for Deck, Pclass, Title, Fare_Bin, Embarked, Adult
Note: Possible values:
      FEATURE   POSSIBLE VALUES
      Deck      A, B, C, D, E, F, G, Z 
      Pclass    1, 2, 3
      Title     lady, master, miss, mr, mrs, officer, sir
      Embarked  S, C, Q
      Adult     Yes, No   # binaryfeature can also be used in one-hot ecoding
# In[617]:


combinedDF = pd.get_dummies(combinedDF, columns=['Deck', 'Pclass', 'Title', 'Embarked', 'Adult'])


# In[618]:


combinedDF.columns


# #### Drop and Reorder column

# In[620]:


combinedDF.drop(['Cabin','Name','Ticket','Parch','SibSp','Sex'], axis=1, inplace=True)


# In[623]:


len(combinedDF.columns)


# In[643]:


# Reorder columns
t_columns = combinedDF.columns.tolist()
t_columns.remove('Survived')
t_columns.sort()
#t_columns.append('Survived') # this adds Survived at the end
t_columns = ['Survived'] + t_columns


# In[644]:


combinedDF = combinedDF[t_columns]


# In[645]:


combinedDF.columns


# In[647]:


combinedDF.info()


# #### SAVE DATAFRAME TO FILE

# In[2]:


import os
processed_data_path = os.path.join(os.path.pardir, 'data', 'processed') #'../data/processed
write_train_data_path = os.path.join(processed_data_path, 'train.csv') #'../data/processed/train.csv' 
combinedDF.loc[combinedDF.Survived != -888].to_csv(write_train_data_path)


# In[651]:


write_test_data_path = os.path.join(processed_data_path, 'test.csv') #'../data/processed/test.csv' 
combinedDF.loc[combinedDF.Survived == -888].to_csv(write_test_data_path) 


# In[653]:


get_ipython().system('jupyter nbconvert --to script Pluralsight_Python_data_science_Abhishek_kumar_2.ipynb')


# ### PLOTTING

# In[659]:


plt.hist(combinedDF.Age, bins=20, color='c')
plt.title('Histogram: Age')
plt.xlabel('Bins of Age')
plt.ylabel('Counts/Frequency')
plt.show()   # to hide extra info


# ### SUBPLOTS

# In[674]:


_ , (ax1, ax2) = plt.subplots(1, 2, figsize = (14,3))
ax1.hist(combinedDF.Age, bins=10, color='g')
ax1.set_title('Histogram: Age')
ax1.set_xlabel('Bins of Age')
ax1.set_ylabel('Counts/Frequency')

ax2.hist(combinedDF.Fare, bins=10, color='tomato')
ax2.set_title('Histogram: Fare')
ax2.set_xlabel('Bins of Fare')
ax2.set_ylabel('Counts/Frequency')
plt.show()


# ##### 5 PLOTS 

# In[680]:


combinedDF.columns


# In[695]:


_ , ax_array = plt.subplots(3, 2) #, figsize=(14,7))
ax_array[0,0].plot(combinedDF.Age)
ax_array[0,1].boxplot(combinedDF.Age)
ax_array[1,0].pie(combinedDF.IsMale)
ax_array[1,1].hist(combinedDF.Age)
ax_array[2,0].plot(combinedDF.Age)

ax_array[2,1].axis('Off')  # to turn offmpty(6th) box

plt.tight_layout()  # to remove overlaps
plt.show()

