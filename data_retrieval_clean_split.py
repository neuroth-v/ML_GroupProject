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



################################################################
path = "data/kickstarter_projects.csv"
SEED = 50
URL1 = 'https://www.kaggle.com/api/v1/datasets/download/ulrikthygepedersen/kickstarter-projects'
URL2 = 'https://www.kaggle.com/api/v1/datasets/download/watts2/glove6b50dtxt'
################################################################

# Import data
req.urlretrieve(URL1, '/data/data.zip')
zipfile.ZipFile('/data/data.zip', 'a').extractall(path='./data/')
req.urlretrieve(URL2, '/data/glove.zip')
zipfile.ZipFile('/data/glove.zip', 'a').extractall(path='./data/')

df_kickstarter = pd.read_csv(path)
df_kickstarter.head(2)

# Converting dates to datetime objects
df_kickstarter['Launched'] = pd.to_datetime(df_kickstarter['Launched'], format='%Y-%m-%d %H:%M:%S')
df_kickstarter['Deadline'] = pd.to_datetime(df_kickstarter['Deadline'], format='%Y-%m-%d')



# Check for duplicates
df_kickstarter.duplicated().sum()



# Log transform the numerical columns. Handle potential errors from log(0)
def log_transform(x):
    return np.log1p(x)

df_kickstarter['log_Goal'] = log_transform(df_kickstarter['Goal'])
df_kickstarter['log_Pledged'] = log_transform(df_kickstarter['Pledged'])
df_kickstarter['log_Backers'] = log_transform(df_kickstarter['Backers'])



# log-transformed distributions look much better and more interpretable. Some observations:
# - left peaks near zero: log-Pledged and log-Backers have sharp peak near 0 (very small or no funding/backers)
# - distribution shapes: log-goal appears roughly symmetrical after transformation, indicating most projects have mid-range funding goals 


# Select numerical columns for correlation analysis
numerical_cols = ['Goal', 'Pledged', 'Backers', 'log_Goal', 'log_Pledged', 'log_Backers']

# Calculate and plot Correlation matrix
correlation_matrix = df_kickstarter[numerical_cols].corr()


# High correlation between 'Pledged' and 'Backers' bzw. 'log_Pledged' and 'log_Backers' - potentially problematic. Options to handle:
# - drop 1 feature, keep the other
# - combine them into a new feature that captures the information from both variables, e.g. 'Pledge_per_Backer' (Ratio to analyze funding efficiency)
# - PCA to reduce multicollinearity and create uncorrelated components from highly correlated features
# - algorithm-specific solutions: dtrees, xgb don't mind correlated features

# #### Relationships with the Target Variable


# Filter for successful/failed campaigns and convert to binary
df_kickstarter_filtered = df_kickstarter[df_kickstarter['State'].isin(['Successful', 'Failed'])]
df_kickstarter_filtered['State_num'] = df_kickstarter_filtered['State'].map({'Successful': 1, 'Failed': 0})



# Summary Statistics:
summary_stats = df_kickstarter_filtered.groupby('State')[numerical_cols].agg(['mean', 'median'])
print(summary_stats)

# #### Distribution of the Target Variable

# In absolute Numbers
state_counts = df_kickstarter_filtered['State'].value_counts()
print(state_counts)

# In Percentage
state_percentage = df_kickstarter_filtered['State'].value_counts(normalize=True) * 100
print(state_percentage)

# %% [markdown]
# **Slight Class Imbalance**
# - while 40:60 isn't severely imbalanced, it can still bias some ML models and lead to suboptimal performance for the minority class
# - metrics: accuracy might appear high simply because the model predicts the majority class most of the time


# Temporal patterns
df_kickstarter_filtered['ProjectDuration'] = (df_kickstarter_filtered['Deadline'] - df_kickstarter['Launched']).dt.days
df_kickstarter_filtered['LaunchYear'] = df_kickstarter_filtered['Launched'].dt.year
df_kickstarter_filtered['LaunchMonth'] = df_kickstarter_filtered['Launched'].dt.month
df_kickstarter_filtered['LaunchDay'] = df_kickstarter_filtered['Launched'].dt.day
df_kickstarter_filtered['DayOfWeek'] = df_kickstarter_filtered['Launched'].dt.dayofweek #0=Monday, 6=Sunday

# Percentage funded, handle 0 cases
def calculate_percentage_funded(row):
    if row['Goal'] == 0:
        if row['Pledged'] > 0:
            return 100  # 100% if Goal is 0 and Pledged > 0
        else:
            return 0    # 0% if Goal and Pledged are 0
    else:
        return (row['Pledged'] / row['Goal']) * 100

# Apply function
df_kickstarter_filtered['PercentageFunded'] = df_kickstarter_filtered.apply(calculate_percentage_funded, axis=1)

# Backers per Dollar pledged, handle 0 cases
def calculate_backers_per_dollar(row):
    if row['Pledged'] == 0:
        return 0 #Handle division by zero
    else:
        return row['Backers'] / row['Pledged']

# Apply function
df_kickstarter_filtered['BackersPerDollar'] = df_kickstarter_filtered.apply(calculate_backers_per_dollar, axis=1)

# Check new features
print(df_kickstarter_filtered[['Goal', 'Pledged', 'Backers', 'PercentageFunded', 'BackersPerDollar']].head())


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






