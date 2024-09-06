# Visualization
import matplotlib.pyplot as plt

# Fate Stay Night Unlimited Frame Works
from sklearn.model_selection import GridSearchCV, train_test_split, PredefinedSplit
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

# Training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
  X, y, test_size=0.2, random_state=69
)

# Validation used for indexes later on
_, Xval, _, yval = train_test_split(
  Xtrain, ytrain, test_size=0.2, random_state=42
)

# ===============================MODEL=========================================

# Creating a custom spliter to know wich data is validation on model fitting
splitIndex = [0 if x in Xval.index else -1 for x in Xtrain.index]
customSplit = PredefinedSplit(splitIndex)

# Parameters that the Grid Search will combine to look for the best model
cvParams = {
  'max_depth': [None, 10, 20, 30],
  'min_samples_leaf': [1, 2, 5],
  'min_samples_split': [2, 5, 10],
  'max_features': [2, 3, 6],
  'n_estimators': [100, 200, 500],
}

# Create pipeline to join the preprocessor with the model search
print('Creating the model...')
pipeline = Pipeline(
  steps=[
    ('preprocessor', preprocessor),
    ('modelsearch', GridSearchCV(
      # Searching for the best Random Forest Classifier
      RandomForestClassifier(random_state=420),
      cvParams, cv=customSplit, n_jobs=-1, verbose=2
    ))
  ]
)

print('Finding best model...')
pipeline.fit(Xtrain, ytrain.values.ravel())

# ==============================RESULTS========================================

print('Best params: ', pipeline['modelsearch'].best_params_)

# Model scores
print('Train score: %f' % pipeline.score(Xtrain, ytrain))
print('Validation score: %f' % pipeline.score(Xval, yval))
print('Test score: %f' % pipeline.score(Xtest, ytest))

trainPred = pipeline.predict(Xtrain)
valPred = pipeline.predict(Xval)
testPred = pipeline.predict(Xtest)

print('Train classification report')
print(classification_report(ytrain, trainPred, zero_division=np.nan))

print('Validation classification report')
print(classification_report(yval, valPred, zero_division=np.nan))

print('Test classification report')
print(classification_report(ytest, testPred, zero_division=np.nan))

ConfusionMatrixDisplay(
  confusion_matrix=confusion_matrix(ytest, testPred), 
  display_labels=pipeline['modelsearch'].best_estimator_.classes_
).plot()
plt.suptitle('Confusion Matrix')
plt.show()
