# Visualization
import matplotlib.pyplot as plt

# Fate Stay Night Unlimited Frame Works
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import classification_report

import numpy as np
# ==========================Fetching and ETL===================================
from ucimlrepo import fetch_ucirepo

print('Fetching data...')
adult = fetch_ucirepo(id=2)

# data (as pandas dataframes)
X = adult.data.features
y = adult.data.targets

# Missing data represents 966 instances from 48842. We can afford to lose some
X = X.dropna()
y = y.loc[X.index]

# Data has a non-significant point "." on some values so we are removing it
y['income'] = y.income.apply(lambda x: x.replace('.', ''))

# Listing categorical columns to one-hot encode them
categCols = [
  'workclass', 
  'education', 
  'marital-status', 
  'occupation', 
  'relationship', 
  'race', 
  'sex', 
  'native-country'
]

# Declare preprocessor tranformers
preprocessor = ColumnTransformer(
  transformers=[
    ('categ', OneHotEncoder(handle_unknown='ignore'), categCols)
  ],
  remainder='passthrough'
)

Xtrain, Xtest, ytrain, ytest = train_test_split(
  X, y, test_size=0.2, random_state=69
)

# ===============================MODEL=========================================

# Create pipeline to join the preprocessor with the classifier
print('Creating the model...')
pipeline = Pipeline(
  steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
      n_estimators=400, max_leaf_nodes=10, n_jobs=-1, random_state=42,
    ))
  ]
)

print('Fitting the model...')
pipeline.fit(Xtrain, ytrain.values.ravel())

# ==============================RESULTS========================================

# Model scores
print('Train score: %f' % pipeline.score(Xtrain, ytrain))
print('Test score: %f' % pipeline.score(Xtest, ytest))

trainPred = pipeline.predict(Xtrain)
testPred = pipeline.predict(Xtest)

print('Train classification report')
print(classification_report(ytrain, trainPred, zero_division=np.nan))

print('Test classification report')
print(classification_report(ytest, testPred, zero_division=np.nan))

ConfusionMatrixDisplay(
  confusion_matrix=confusion_matrix(ytest, testPred), 
  display_labels=pipeline['classifier'].classes_
).plot()
plt.suptitle('Confusion Matrix')
plt.show()
