######################################################################################################################
# Output of this script - JOINED MODEL final_joined.sav (in /models folder)  
#
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
from sklearn.pipeline import Pipeline
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

#####################################################################################################################################
# Constants

#####################################################################################################################################


# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Preprocess text: remove punctuation, convert to lowercase, remove stopwords
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = text.lower()
    text = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words]
    stemmer = PorterStemmer()
    text = [stemmer.stem(word) for word in text]
    return " ".join(text)

# Apply preprocessing to 'Name' column
df_kickstarter_filtered['cleaned_name'] = df_kickstarter_filtered['Name'].apply(preprocess_text)

# Load GloVe embeddings in batches
glove_file = 'data/glove.6B.50d.txt'
glove_model = {}

with open(glove_file, 'r', encoding='utf-8') as f: #Open the file
    for line in f:
        parts = line.strip().split()
        word = parts[0]
        vector = np.array(parts[1:], dtype=float) #Convert vector string parts to floats
        glove_model[word] = vector #Store the word and vector in the dictionary

print(f"GloVe model '{glove_file}' loaded successfully (in batches).")

# Use glove_model dictionary to create document vectors
def create_document_vector(title, glove_model):
    words = title.split()
    vectors = []
    for word in words:
        if word in glove_model: #Check if the word is actually in the dict
            vectors.append(glove_model[word])
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(50) #50 because we use the 50-dimensional GloVe embeddings

# Apply function
df_kickstarter_filtered['glove_vector'] = df_kickstarter_filtered['cleaned_name'].apply(lambda x: create_document_vector(x, glove_model))

# Check and verify embedding
print(df_kickstarter_filtered[['cleaned_name', 'glove_vector']].head())

# Zero vector analysis 
def is_zero_vector(vector):
    return all(v == 0 for v in vector)

zero_vector_count = df_kickstarter_filtered['glove_vector'].apply(is_zero_vector).sum()
total_vectors = len(df_kickstarter_filtered)
zero_vector_percentage = (zero_vector_count / total_vectors) * 100

print(f"Number of zero vectors: {zero_vector_count}/{total_vectors}")
print(f"Percentage of zero vectors: {zero_vector_percentage:.2f}%")

# Quality Control: Cosine Similarity Analysis
def calculate_cosine_similarity(title1, title2, glove_model):
    vec1 = create_document_vector(title1, glove_model)
    vec2 = create_document_vector(title2, glove_model)
    similarity = cosine_similarity([vec1], [vec2])[0][0]  #Get the single similarity score
    return similarity

# Cluster to find similar project names
glove_matrix = np.array(df_kickstarter_filtered['glove_vector'].tolist())

# Scale the data (important for KMeans)
scaler = StandardScaler()
glove_matrix_scaled = scaler.fit_transform(glove_matrix)

# Choose the number of clusters (k) – experiment to find a good value
k = 10  #Start with a small number and experiment. You can use the elbow method to determine k.
kmeans = KMeans(n_clusters=k, random_state=42) #setting random state for reproducibility.
kmeans.fit(glove_matrix_scaled)

df_kickstarter_filtered['cluster'] = kmeans.labels_

# Find similar titles within the same cluster.
for i in range(k):
    cluster_df = df_kickstarter_filtered[df_kickstarter_filtered['cluster'] == i]
    print(f"Cluster {i}:")
    print(cluster_df[['Name', 'cleaned_name']].head()) #Show top few titles in each cluster

# Pick example pairs for cosine similarity analysis. 
# Strategy: similar pairs (within-cluster), dissimilar pairs (between-clusters), and edge cases (diverse titles)
example_pairs = [
    # similar pairs within clusters
    ("offlin wikipedia iphon app", "icon iphon app"), #Cluster 3
    ("crystal antler untitl movi", "might becom movi"), #Cluster 1 and 7 - interesting
    ("hand made guitar pick", "new kitchen tool"), #Cluster 0 and 2 - dissimilar
    
    # dissimilar pairs between clusters
    ("offlin wikipedia iphon app", "hand made guitar pick"),
    ("web site short horror film", "mr squiggl"),
    ("logic guess pictur 2nd horror movi ", "kicey iceland")
]

for example_pair in example_pairs:
    similarity = calculate_cosine_similarity(example_pair[0], example_pair[1], glove_model)
    print(f"Cosine similarity between '{example_pair[0]}' and '{example_pair[1]}': {similarity:.3f}")

# #### Summary Interpretation:
# - GloVe embedding seems reasonably effective in capturing semantic relationships, particularly for closely related titles
# - however: 'hand made guitar pick' and 'new kitchen tool' share high similarity as well - maybe due to limitations of the GloVe model (trained on general corpus of words, not specifically project titles)

# # One-hot Encoding categorical features

# Categorical columns
categorical_cols = ['Category', 'Subcategory', 'Country', 'DayOfWeek', 'LaunchMonth', 'LaunchYear']

# Create column transformer
ct = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ],
    remainder='passthrough' #keep other columns as they are
)

# Apply one-hot encoding
encoded_data = ct.fit_transform(df_kickstarter_filtered)

encoded_df = pd.DataFrame(encoded_data, columns=ct.get_feature_names_out())
encoded_df['Name'] = df_kickstarter_filtered['Name'] #add it back manually

#Convert back to DataFrame
encoded_df = pd.DataFrame(encoded_data, columns=ct.get_feature_names_out())

#Verification: Check the shape of the DataFrame
print("Original DataFrame shape:", df_kickstarter_filtered.shape)
print("Encoded DataFrame shape:", encoded_df.shape)

final_df = encoded_df['remainder__State_num']

# ## Splitting data for testing 
X_train, X_test, y_train, y_test = train_test_split(final_df.drop('remainder__State_num', axis=1), final_df.remainder__State_num, train_size=0.8, stratify=final_df.remainder__State_num, random_state=SEED)
X_train.to_csv('./data/X_train.csv',index=False)
X_test.to_csv('./data/X_test.csv',index=False)
y_train.to_csv('./data/y_train.csv',index=False)
y_test.to_csv('./data/y_test.csv',index=False)

########################################################################################################################


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
    df['lin_duration_days'] = df['deadline'] - df['launched']
    df['lin_duration_days'] = max_frac(df['lin_duration_days'].dt.days.astype('int'))


# Splitting ("dymming") and dumping launch dates:
#           LAUNCH DATE
    df['launched_day_frac'] = df['launched'].dt.day/31
    df['launched_month_frac'] = df['launched'].dt.month/12
    df['launched_year_frac'] = df['launched'].dt.year/2024
    df['launched_dow'] = df['launched'].dt.day_name()
    df = df.join(pd.get_dummies(df['launched_dow'], prefix='launched', prefix_sep='_', drop_first = False, dtype=float)).copy()
    least_common = pd.DataFrame(df['launched_dow'].value_counts(sort=True, ascending=True)).index.tolist()[0]
    df = df.drop('launched_'+least_common, axis=1)
    df = df.drop(['launched_dow', 'launched'], axis=1)
#           DEADLINE DATE
    df['deadline_day_frac'] = df['deadline'].dt.day/31
    df['deadline_month_frac'] = df['deadline'].dt.month/12
    df['deadline_year_frac'] = df['deadline'].dt.year/2024
    df['deadline_dow'] = df['deadline'].dt.day_name()
    df = df.join(pd.get_dummies(df['deadline_dow'], prefix='deadline', prefix_sep='_', drop_first = False, dtype=float)).copy()
    least_common = pd.DataFrame(df['deadline_dow'].value_counts(sort=True, ascending=True)).index.tolist()[0]
    df = df.drop('deadline_'+least_common, axis=1)
    df = df.drop(['deadline', 'deadline_dow'], axis=1)
    
# calculating dol_p_back:
    #df = calc_dol_p_back(df)
  
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
# Vectorizing name column:



    
    return df
#####################################################################################################################################################




