######################################################################################################################
# Output of this script (will be) - MODEL (in .sav format, stored in /models folder) 
#
# Base and Sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import urllib.request as req
import zipfile

# EDA
import missingno as msno

# Visualisation
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.tree import plot_tree

# Text-Based Analyses
import gensim.downloader
from gensim.models import KeyedVectors
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from sklearn.metrics.pairwise import cosine_similarity

# One-hot Encoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Voting
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier

# Classifiers
from sklearn.tree import DecisionTreeClassifier                                     
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV                
from sklearn.linear_model import SGDClassifier                                      
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors                                      
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC                                                        
from sklearn.ensemble import AdaBoostClassifier 

# Regressors
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Clustering
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import scipy.cluster.hierarchy as sch  

# Scalers and Transformers
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline                                               
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from sklearn.impute import SimpleImputer

# Metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, roc_curve, roc_auc_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy.stats import zscore, boxcox
from sklearn.model_selection import cross_val_score



# Main code starts below:
#####################################################################################################################################

# Loading splits for processing
X_train = pd.read_csv('./data/X_train.csv')
X_test = pd.read_csv('./data/X_test.csv')
y_train = pd.read_csv('./data/y_train.csv')
y_test = pd.read_csv('./data/y_test.csv')


def calc_dol_p_back(df:pd.DataFrame)-> pd.DataFrame:
    df['backers'] = df['backers'] + np.nextafter(0,1)
    df['log_dol_p_back'] = df['pledged']/df['backers']
    df['backers'] = df['backers'].astype('int') 
    return df

def max_frac(ds:pd.Series)-> pd.Series:
    if ds.max() != 0:
        ds = ds/ds.max()
    return ds

def log10p1_of_val(val):
    return np.log10(val+1)


def pre_processing(df:pd.DataFrame)-> pd.DataFrame:
# Formating dates
    df['launched'] = pd.to_datetime(df['launched'], format='%Y-%m-%d %H:%M:%S')
    df['deadline'] = pd.to_datetime(df['deadline'], format='%Y-%m-%d')
    df['lin_duration_days'] = df['deadline'] - X_train['launched']
    df['lin_duration_days'] = max_frac(df['lin_duration_days'].dt.days.astype('int'))


# Splitting ("dymming") and dumping launch dates:
#           LAUNCH DATE
    df['launched_day'] = df['launched'].dt.day
    df['launched_month'] = df['launched'].dt.month
    df['launched_year'] = df['launched'].dt.year
    df['launched_dow'] = df['launched'].dt.day_name()
    df = df.join(pd.get_dummies(df['launched_dow'], prefix='launched', prefix_sep='_', drop_first = False, dtype=float)).copy()
    least_common = pd.DataFrame(df['launched_dow'].value_counts(sort=True, ascending=True)).index.tolist()[0]
    df = df.drop('launched_'+least_common, axis=1)
    df = df.drop(['launched_dow', 'launched'], axis=1)
#           DEADLINE DATE
    df['deadline_day'] = df['deadline'].dt.day
    df['deadline_month'] = df['deadline'].dt.month
    df['deadline_year'] = df['deadline'].dt.year
    df['deadline_dow'] = df['deadline'].dt.day_name()
    df = df.join(pd.get_dummies(df['deadline_dow'], prefix='deadline', prefix_sep='_', drop_first = False, dtype=float)).copy()
    least_common = pd.DataFrame(df['deadline_dow'].value_counts(sort=True, ascending=True)).index.tolist()[0]
    df = df.drop('deadline_'+least_common, axis=1)
    df = df.drop(['deadline', 'deadline_dow'], axis=1)
    
# calculating dol_p_back:
    df = calc_dol_p_back(df)
  
# Logarithmizing
    df['goal_lin'] = max_frac(df['goal'])
    df['pledged_lin'] = max_frac(df['pledged'])
    df['backers_lin'] = max_frac(df['backers'])
#    df['pledged_log10'] = df['pledged'].apply(lambda x: log10p1_of_val(x))
#    df['backers_log10'] = df['backers'].apply(lambda x: log10p1_of_val(x))

    df = df.drop(['goal', 'pledged', 'backers'], axis=1).copy()


# Splitting ("dymming") and object columns:
    obs = []
    for val in df.drop('name', axis=1).columns.tolist():
        if df[val].dtype.name == 'object':
            obs.append(val)
    for val in obs:
        least_common = pd.DataFrame(df[val].value_counts(sort=True, ascending=True)).index.tolist()[0]
        df = df.join(pd.get_dummies(df[val], prefix=val, prefix_sep='_', drop_first = False, dtype=int))
        df = df.drop(val+'_'+least_common, axis=1)
    df = df.drop(obs, axis = 1)

    return df


X_train = pre_processing(X_train)


#################################################################################################






