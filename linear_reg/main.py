import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from ucimlrepo import fetch_ucirepo

from mysklearn import StandardScaler, LinearRegression, SplitDataset

def GetSamplesAndYs(df: pd.DataFrame):
  samples = df.drop(['murders'], axis=1)
  y = df[['murders']]
  
  return (samples, y)

print('Fetching dataset')
CrimeRepo = fetch_ucirepo(id=211) 
  
# data (as pandas dataframes) 
features = CrimeRepo.data.features
targets = CrimeRepo.data.targets

print('Processing dataset')
X = features.dropna(axis=1)

StateEncoder = {x: i for i, x in enumerate(X['State'].unique())}

X.loc[:, 'State'] = X['State'].map(StateEncoder).copy()

df = pd.concat([X, targets[['murders']]], axis=1)

train, val, test = SplitDataset(df)

Scaler = StandardScaler(train)

train = Scaler(train)
val = Scaler(val)
test = Scaler(test)

model = LinearRegression()

alpha = 0.01

trainSamples, trainY = GetSamplesAndYs(train)
valSamples, valY = GetSamplesAndYs(val)

print('Start Model Fitting')
model.fit(trainSamples, trainY, valSamples, valY, alpha)
print('Finished Fitting')

plt.clf()
sns.lineplot(model.valErrors)
ax = sns.lineplot(model.trainErrors)
ax.lines[1].set_linestyle('--')
ax.legend(['Validation Errors', 'Train Errors'])
plt.show()

testSamples, testY = GetSamplesAndYs(test)

r2 = model.score(testSamples, testY)

print('Error (r2):', r2)

sns.color_palette('tab10', as_cmap=True)
sns.set_palette('tab10')
preds = model.predict(testSamples)

plt.clf()
sns.scatterplot(x=np.arange(preds.shape[0]), y=preds.flatten())
ax = sns.scatterplot(x=np.arange(preds.shape[0]), y=testY.to_numpy().flatten())
ax.legend(['Predicted', 'Real'])
plt.show()

