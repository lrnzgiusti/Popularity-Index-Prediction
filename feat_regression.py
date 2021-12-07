#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 16:40:22 2021

@author: ince
"""
from supervised.automl import AutoML
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import StandardScaler

plt.style.use("ggplot")
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from yellowbrick.target import FeatureCorrelation
from scipy.stats import norm
from scipy import stats

from wordcloud import WordCloud

import warnings
from yellowbrick.features import JointPlotVisualizer


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import plotly.express as px
warnings.filterwarnings("ignore")
color = sns.color_palette()
color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]
import plotly.io as pio
pio.renderers.default = 'sphinx_gallery' # or 'notebook' or 'colab' or 'jupyterlab'


df = pd.read_csv("./data/features/features.csv")\
       .drop(columns=['title'])\
       .sample(frac=1)\
       .reset_index(drop=True)

df = df.apply(pd.to_numeric)
feat_cols=[ 'happiness', 'danceability', 'energy', 'acousticness',
           'instrumentalness', 'speechiness']

df['valence'] = df['happiness']

feat_cols=[ 'valence', 'danceability', 'energy', 'acousticness',
           'instrumentalness', 'speechiness']
df.drop(columns=['happiness'], inplace=True)

import pickle
from sklearn.decomposition import PCA

new_conv_feat = []

L = pickle.load(open(r"/Users/ince/Desktop/base/conv_feat.pkl", "rb"))

for l in L:
    new_conv_feat.append(l.cpu().detach().numpy().flatten())

A = np.array(new_conv_feat)[:347]
pca = PCA(n_components=16)
A_pca = pca.fit_transform(A)
X, y = df[feat_cols], df['popularity']

X = np.hstack((X,A_pca))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    

automl = AutoML(mode="Perform", total_time_limit=5*60 , explain_level=2 )  
automl.fit(X_train, y_train)
