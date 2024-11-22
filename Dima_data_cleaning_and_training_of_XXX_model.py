# BASE
import pandas as pd
import numpy as np
import sklearn
from sklearn import set_config
import pickle

#VISUALIZATION
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.tree import plot_tree


#??????????????????????? WHAT ARE YOU????????????????????????
from scipy.stats import mode                                                        #?????????????????????????????????
from sklearn.datasets import load_digits                                            #?????????????????????????????
from sklearn.decomposition import PCA                                               #??????????????????????????
from sklearn.manifold import TSNE                                                   #????????????????????????????
from sklearn.datasets import make_classification                                    #??????????????????????????
from scipy.special import expit                                                     #???????????????????????????


#VOTING
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier

from xgboost import XGBClassifier


#CLASIFIERS
from sklearn.tree import DecisionTreeClassifier                                     
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV                # search of the best params for random_forrest
from sklearn.linear_model import SGDClassifier                                      #????????????????????????
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors                                      #????????????????????????
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC                                                         #???????????
from sklearn.ensemble import AdaBoostClassifier                                     #?????????????




#REGRESSORS
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import ElasticNet

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor 


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor





#CLASTERING
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import scipy.cluster.hierarchy as sch                                               #????????????????????????????




# SCALERS and TRANSFORMATION
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import FunctionTransformer

from sklearn.pipeline import Pipeline                                               # pipeline function for transformers
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from sklearn.impute import SimpleImputer




# metrics and processing 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, roc_curve, roc_auc_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from sklearn.model_selection import train_test_split
from scipy.stats import zscore, boxcox
from sklearn.model_selection import cross_val_score                                 #??????????????????????????



# EDA data treatment
import missingno


# other (mainly system libs)
import warnings
import sys
from io import StringIO
import time
import os
import shutil
from dotenv import load_dotenv
from sqlalchemy import create_engine
import itertools
from collections import Counter
import urllib.request as req
import zipfile


#####################################################
# Constants
SEED = 50
URL1 = 'https://www.kaggle.com/api/v1/datasets/download/ulrikthygepedersen/kickstarter-projects'
URL2 = 'https://www.kaggle.com/api/v1/datasets/download/watts2/glove6b50dtxt'
################################################################

# Downloading data
req.urlretrieve(URL1, '/data/data.zip')
zipfile.ZipFile('/data/data.zip', 'a').extractall(path='./data/')
req.urlretrieve(URL2, '/data/glove.zip')
zipfile.ZipFile('/data/glove.zip', 'a').extractall(path='./data/')

# Processing downloads
df = pd.read_csv('./data/kickstarter_projects.csv')
print(df.info())
print(df.describe())

df.columns = [val.strip().replace(' ','_').lower() for val in df.columns.tolist()]
df.drop('id', axis=1, inplace=True)
df['launched'] = pd.to_datetime(df['launched'], format='%Y-%m-%d %H:%M:%S')
df['deadline'] = pd.to_datetime(df['deadline'], format='%Y-%m-%d')
df = df[(df['state'] == 'Successful') | (df['state'] == 'Failed')]
df['state'] = df['state'].map({'Successful':1 , 'Failed': 0})


# ## Splitting data for testing 
X_train, X_test, y_train, y_test = train_test_split(df.drop('state', axis=1), df.state, train_size=0.8, stratify=df.state, random_state=SEED)
X_train.to_csv('./data/X_train.csv',index=False)
X_test.to_csv('./data/X_test.csv',index=False)
y_train.to_csv('./data/y_train.csv',index=False)
y_test.to_csv('./data/y_test.csv',index=False)
#####################################################################################################################################
# Loading splits and process them:

X_train = pd.read_csv('./data/X_train.csv')
X_test = pd.read_csv('./data/X_test.csv')
y_train = pd.read_csv('./data/y_train.csv')
y_test = pd.read_csv('./data/y_test.csv')

def log10p1_of_val(val):
    return np.log10(val+1)




def calc_dol_p_back(df:pd.DataFrame)-> pd.DataFrame:
    df['backers'] = df['backers'] + np.nextafter(0,1)
    df['log_dol_p_back'] = df['pledged']/df['backers']
    df['backers'] = df['backers'].astype('int')
       
    return df




def pre_processing(df:pd.DataFrame)-> pd.DataFrame:
# Formating dates
    df['launched'] = pd.to_datetime(df['launched'], format='%Y-%m-%d %H:%M:%S')
    df['deadline'] = pd.to_datetime(df['deadline'], format='%Y-%m-%d')
    df['duration_days'] = df['deadline'] - X_train['launched']
    df['duration_days'] = df['duration_days'].dt.days


# Splitting and dumping launch dates:
    df['launched_day'] = df['launched'].dt.day
    df['launched_month'] = df['launched'].dt.month
    df['launched_year'] = df['launched'].dt.year
    df['launched_dow'] = df['launched'].dt.day_name()
    df = df.join(pd.get_dummies(df['launched_dow'], prefix='launched_', drop_first = True, dtype=float))

    df['deadline_day'] = df['deadline'].dt.day
    df['deadline_month'] = df['deadline'].dt.month
    df['deadline_year'] = df['deadline'].dt.year
    df['deadline_dow'] = df['deadline'].dt.day_name()
    df = df.join(pd.get_dummies(df['deadline_dow'], prefix='deadline_', drop_first = True, dtype=float))

    df = df.drop(['launched','deadline','launched_dow','deadline_dow'], axis=1)
    
# calculating dol_p_back:
    df = calc_dol_p_back(df)
  
# Logarithmizing
    df['goal_log10'] = df['goal'].apply(lambda x: log10p1_of_val(x))
    df['pledged_log10'] = df['pledged'].apply(lambda x: log10p1_of_val(x))
    df['backers_log10'] = df['backers'].apply(lambda x: log10p1_of_val(x))

    df = df.drop(['goal', 'pledged', 'backers'], axis=1)


    return df


pre_processing(X_train).info()

#################################################################################################






