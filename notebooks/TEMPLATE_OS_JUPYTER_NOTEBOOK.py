
# coding: utf-8

# ### Notes:

# In[ ]:


NOTES:
# • Python Distributions for Data Science:
#     ->Ananconda
#     ->Enthought Canopy

# • Jupyter Notebook:
# 		○ Combine code, text, images, videos
# 		○ Any browser
# 		○ Supports below kernels:
# 			§ Python
# 			§ R
# 			§ Julia
# 			§ Scala
# 		○   Viewer: nbviewer(no browser needed)(in github)
# 		○ Export to pdf, HTML, etc

# • Jupyter notebook [--port <port_no>] #default:8888
#                    [--no-browser] # useful for cloud


# ### IMAGE

# In[ ]:


from IPython.display import Image 
#Image(filename='images/extracting_data_from_db_template.png', height=300, width=400)


# ### Import Youtube videos

# In[ ]:


from IPython.display import YouTubeVideo
YouTubeVideo("pN4HqWRybwk")


# ### Warnings

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# ### CONVERT .ipynb to .py

# In[ ]:


get_ipython().system('jupyter nbconvert --to script Pluralsight_Python_data_science_Abhishek_kumar_2.ipynb')
get_ipython().system('jupyter nbconvert --to script pyspark_baled_linechart_sql_test.ipynb')
get_ipython().system('jupyter nbconvert --to script UDEMY_DEEP_LEARNING_A-Z_ANN_KIRILL_2.ipynb')


# In[ ]:


'''
!conda install -y nbconvert
Solving environment: done

## Package Plan ##

  environment location: /anaconda3

  added / updated specs: 
    - nbconvert


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    defusedxml-0.5.0           |           py36_1          29 KB
    openssl-1.0.2p             |       h1de35cc_0         3.4 MB
    nbconvert-5.4.0            |           py36_1         416 KB
    conda-4.5.11               |           py36_0         1.0 MB
    certifi-2018.8.24          |           py36_1         139 KB
    ------------------------------------------------------------
                                           Total:         5.0 MB

The following NEW packages will be INSTALLED:

    defusedxml:      0.5.0-py36_1                    

The following packages will be UPDATED:

    certifi:         2018.8.24-py36_1     conda-forge --> 2018.8.24-py36_1 
    conda:           4.5.11-py36_0        conda-forge --> 4.5.11-py36_0    
    nbconvert:       5.3.1-py36h810822e_0             --> 5.4.0-py36_1     
    openssl:         1.0.2p-h470a237_0    conda-forge --> 1.0.2p-h1de35cc_0

The following packages will be DOWNGRADED:

    ca-certificates: 2018.8.24-ha4d7672_0 conda-forge --> 2018.03.07-0     


Downloading and Extracting Packages
defusedxml-0.5.0     | 29 KB     | ##################################### | 100% 
openssl-1.0.2p       | 3.4 MB    | ##################################### | 100% 
nbconvert-5.4.0      | 416 KB    | ##################################### | 100% 
conda-4.5.11         | 1.0 MB    | ##################################### | 100% 
certifi-2018.8.24    | 139 KB    | ##################################### | 100% 
Preparing transaction: done
Verifying transaction: done
Executing transaction: done

​
'''


# ### PATH

# In[ ]:


import os
raw_file_path = os.path.join(os.path.pardir, 'data', 'raw') 

consolidatemill_file_path = os.path.join(raw_file_path, 'consolidate_mill.csv')
humidity_data_path = os.path.join(raw_file_path, 'dc_humidity.csv')
date_data_path = os.path.join(raw_file_path, 'date_data.csv')

location_division_mapping_file = os.path.join(os.path.pardir, 'data', 'raw', 'Location_Division_Mapping.csv') 


# ### RUN .PY SCRIPT FROM JUPYTER

# In[ ]:


import os
get_data_processing_file = os.path.join(os.path.pardir, 'src', 'data', 'Titanic_processing_script.py')
 
get_ipython().system('python $get_data_processing_file')
# Output gets written below this cell


# ### OS INFO

# In[ ]:


get_ipython().system('pwd')
get_ipython().system('python --version')


# In[4]:


get_ipython().run_line_magic('env', '')


# In[2]:


get_ipython().system('pip list | grep ipython ')


# In[3]:


get_ipython().system('conda list | grep ipython ')


# In[1]:


import os
get_ipython().system(' ls $os.path.pardir')


# In[2]:


get_ipython().system(' ls ../$os.path.pardir')


# In[ ]:


import os
os.curdir

os.pardir

os.path.dirname(os.pardir)

os.getcwd()

os.path.abspath(__name__)

os.getegid()

os.getenv('SPARK_HOME')

os.environ

my_folder = get_ipython().getoutput('pwd')
my_folder

get_ipython().system('pwd')


# ### Housekeeping Global file

# In[3]:


housekeeping_file = os.path.join(os.pardir , 'src' , 'scripts', 'housekeeping.py')
housekeeping_file


# In[5]:


get_ipython().system('cat $housekeeping_file  ')


# In[ ]:


get_ipython().run_line_magic('load', '../src/scripts/housekeeping.py')


# In[ ]:


# %load ../src/scripts/housekeeping.py
#Housekeeping\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os

from IPython.display import Image
from IPython.core.display import HTML 
#Image(filename='images/extracting_data_from_db_template.png', height=300, width=400)}


# ### GLOB in python

# In[ ]:


def processAll(path):
    
    allDF = pd.DataFrame()
    folder_names = glob.glob(path + '\\*') 
    for f in folder_names:
        df = processFolder(f)
        allDF = allDF.append(df, ignore_index=True, sort=True)
    return allDF

def processFolder(folder_name): 
    partialDF = pd.DataFrame()
    current_files = folder_name  + '\*.xls'
    file_names = glob.glob(current_files)  
    for f in file_names:
        df = pd.read_excel(f)
        partialDF = partialDF.append(df,ignore_index=True, sort=True)
    return partialDF

import glob
# STEP 1: Process all folders
raw_file_path = os.path.join(os.path.pardir, 'data', 'raw', 'mill_data_all') 
finalDF = processAll(raw_file_path) 


# ### EXECUTE USNG .PY FILE

# #Usage: C:\OR\PROJECTS\BALER\BALER_JUPYTER_PROJECT\baler\notebooks>python Combine_Mill_Data.py <root_path> | default

# ### PRINT

# In[ ]:


# Define some helpful functions
def printDf(sprkDF): 
    newdf = sprkDF.toPandas()
    from IPython.display import display, HTML
    return HTML(newdf.to_html())


# ## MARKDOWN

# ### BLOCK QUOTES

# >lock quote blah blah bblah blah block quote blah blah block quote blah blah block quote blah blah block quote blah blah 
# lock quote blah blah bblah blah block quote blah blah block quote blah blah block quote blah blah block quote blah blah block quote blah blah block quote blah blah block quote blah blah block quote blah blah block quote

# ***

# ### HORIZONTAL LINE

# #***

# ***

# ### NESTED BLOCKQUOTES

# >blah blah block quote blah blah block quote blah blah block  
# block quote blah blah block block quote blah blah block  
# >>quote blah blah block quote blah blah  
# block block quote blah blah block   
# >>>quote blah blah block quote blah blah block quote blah blah block quote

# ### ORDERED LIST
0. Fruit:
    6. Pears
    0. Peaches
    3. Plums
    4. Apples 
        2. Granny Smith 
        7. Gala
    * Oranges
    - Berries 
        8. Strawberries 
        + Blueberries
        * Raspberries
    - Bananas
9. Bread:
    9. Whole Wheat
        0. With oats on crust
        0. Without oats on crust
    0. Rye 
    0. White
0. Dairy:
    0. Milk
        0. Whole
        0. Skim
    0. Cheese
        0. Wisconsin Cheddar
        0. Pepper Jack
# 0. Fruit:
#     6. Pears
#     0. Peaches
#     3. Plums
#     4. Apples 
#         2. Granny Smith 
#         7. Gala
#     * Oranges
#     - Berries 
#         8. Strawberries 
#         + Blueberries
#         * Raspberries
#     - Bananas
# 9. Bread:
#     9. Whole Wheat
#         0. With oats on crust
#         0. Without oats on crust
#     0. Rye 
#     0. White
# 0. Dairy:
#     0. Milk
#         0. Whole
#         0. Skim
#     0. Cheese
#         0. Wisconsin Cheddar
#         0. Pepper Jack

# ---

# ---

# ### LINKS

# #### Standard Link:

# [click this link](http://en.wikipedia.org)

# #### Standard Links with mouse over title:

# [click this link](http://en.wikipedia.org "Wikipedia")

# #### Automatic link:

# http://en.wikipedia.org

# ### TABLES
Header|Header|Header|Header
-|-|-|-
Cell|Cell|Cell|Cell
Cell|Cell|Cell|Cell
Cell|Cell|Cell|Cell
Cell|Cell|Cell|Cell
# Header|Header|Header|Header
# -|-|-|-
# Cell|Cell|Cell|Cell
# Cell|Cell|Cell|Cell
# Cell|Cell|Cell|Cell
# Cell|Cell|Cell|Cell

# ### DECORATIONS ON TEXT

# * ITALICS
# 
# *Italics*
# 
# * BOLD
# 
# **Bold**
# 
# * INDENT
# 
# $Indent1$
# $$Indent2$$
# 
# * LEVELS
# > LEVEL1
# >> LEVEL2
# 
# * OTHER CHARACTERS
# 
# Ampersand &amp; Ampersand
# 
# &lt; angle brackets &gt;
# 
# &quot; quotes &quot;
# 

# ### IMAGES

# ![It doesn't matter what you write here](http://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/South_African_Giraffe,_head.jpg/877px-South_African_Giraffe,_head.jpg "Picture of a Giraffe)

# ### LaTeX Math

# $z=\dfrac{2x}{3y}$

# $F(k) = \int_{-\infty}^{\infty} f(x) e^{2\pi i k} dx$

# ### COLOR

# This is <font color='red'>RED</font> color
# 
# 
# $\color{red}{\text{RED}}$
# 
# <p style="font-family: Arial; font-size:1.4em;color:gold;"> Golden </p>
# 
# <p style="color:red;">ERROR: Setting focus didn't work for me when I tried from jupyter. However it worked well when I ran it from the terminal</p>

# #### > INDENT
# > INDENT

# #### - BULLET ITEM
# - BULLET ITEM

# ####  -  ITEM2
#  -  ITEM2

# #### * ITEM3
# * ITEM3

# #### * MAIN BULLET
# ####   * SUB_BULLET
# * MAIN BULLET
#    * SUB_BULLET    

# ####
# #### 1. NUMBERED ITEM1
# #### 1. NUMBERED ITEM2
# #### 1. NUMBERED ITEM3
# #### 1. NUMBERED ITEM4
# ####
# 
# 1. NUMBERED ITEM1
# 1. NUMBERED ITEM2
# 1. NUMBERED ITEM3
# 1. NUMBERED ITEM4

# <div class="alert alert-block alert-danger"> This is <b>RED</b> colored text </div>

# <div class="alert alert-block alert-danger">
# <b>Just don't:</b> In general, avoid the red boxes. These should only be
# used for actions that might cause data loss or another major issue.
# </div>

# ***

# __[Google web site]http://google.com__

# ### Line Numbers

# In[1]:


#Line numbers keyboard shortcut: [ESC] + L


# ### Magic Functions

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Use % for single line
# Use %% for multi line


# In[13]:


name = 'GANESH'


# In[26]:


get_ipython().run_cell_magic('writefile', 'test.txt', 'this is written from jupyter notebook %name')


# In[27]:


get_ipython().system('cat test.txt')


# In[9]:


get_ipython().run_cell_magic('HTML', '', '<p> List1 </i>\n<p> List2 </i>')


# In[10]:


get_ipython().run_cell_magic('latex', '', '\\begin{align}\nGradient : \\nabla J = - 2H^T (Y_HW)\n\\end{align}')


# In[28]:


get_ipython().run_cell_magic('timeit', 'x = range(1000)', 'max(x)')


# ### HTML RENDERING

# In[33]:


html_string = '''
<!doctype html>
<html lang=\"en\">
    "<head>
          <title>Doing Data Science With Python</title>
    </head>
    <body>
          <h1 style=\"color:#F15B2A;\">Doing Data Science With Python</h1>
          <p id=\"author\">Author : Abhishek Kumar</p>
    <p id=\"description\">This course will help you to perform various data science activities using python.</p>
          
          <h3 style=\"color:#404040\">Modules</h3>
          <table id=\"module\" style=\"width:100%\">
              <tr>
                <th>Title</th>
                <th>Duration (In Minutes)</th> 
              </tr>
              <tr>
                <td>Getting Started</td>
                <td>20</td> 
              </tr>
              <tr>
                <td>Setting up the Environment</td>
                <td>40</td> 
              </tr>
              <tr>
                <td>Extracting Data</td>
                <td>35</td> 
              </tr>
              <tr>
                <td>Building Predictive Model</td>
                <td>30</td> 
              </tr>
          </table>
    </body>
    </html>
    '''


# In[34]:


from IPython.core.display import display, HTML
display(HTML(html_string))


# ### Convert notebook to script

# In[39]:


get_ipython().system('jupyter nbconvert TEMPLATE_OS_JUPYTER_NOTEBOOK.ipynb')


# In[ ]:


get_ipython().system('jupyter nbconvert -TEMPLATE_OS_JUPYTER_NOTEBOOK.ipynb')

