# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 15:40:15 2020

Author: Pipe Galera

Project: Introduction to KNN

Goal: classify cars in 4 categories based upon certain features.

"""

%reset

# Import modules 

import pandas as pd
import numpy as np
import zipfile
import os
from sklearn import preprocessing, linear_model
from sklearn.neighbors import KNeighborsClassifier

# Read the data

os.chdir("C:/Users/fgm.si/Documents/GitHub/ML-AI-Megacourse/2. K-Nearest Neighbors")
data = pd.read_csv("raw_data/car.data")

# Pre-Analysis and cleanning

data.isnull().sum()
data.dtypes

le = preprocessing.OneHotEncoder()



