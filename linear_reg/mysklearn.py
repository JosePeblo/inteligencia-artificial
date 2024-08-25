import numpy as np
import pandas as pd

def h(params: np.ndarray, samples: np.ndarray):
  return samples @ params
  
def GD(params: np.ndarray, samples: np.ndarray, y: np.ndarray, alpha: float):
  err = h(params, samples) - y
  gradient = (samples.T @ err) / samples.shape[0]
  return params - alpha * gradient

def loss(params: np.ndarray, samples: np.ndarray, y: np.ndarray):
  hyp = h(params, samples)
  return np.square(hyp - y).mean()

class LinearRegression:
  def __init__(self):
    self.params = None
    self.trainErrors = []
    self.valErrors = []

  def __dataSetup__(self, samples: pd.DataFrame, y: pd.DataFrame):
    Samples = samples.to_numpy()
    Samples = np.hstack((np.ones((Samples.shape[0], 1)), Samples))
    Y = y.to_numpy()
    return (Samples, Y)

  def fit(
      self, 
      trainSamples: pd.DataFrame, 
      trainY: pd.DataFrame, 
      valSamples: pd.DataFrame, 
      valY: pd.DataFrame,
      alpha = 0.01
    ):
    
    self.trainErrors = []
    self.valErrors = []
    
    TrainSamples, TrainY = self.__dataSetup__(trainSamples, trainY)
    ValSamples, ValY = self.__dataSetup__(valSamples, valY)

    self.params = np.random.random((TrainSamples.shape[1], 1))
    self.params[0][0] = 1

    for epoch in range(40000):
      oldParams = self.params.copy()

      self.params = GD(self.params, TrainSamples, TrainY, alpha)

      trainErr = loss(self.params, TrainSamples, TrainY)
      valErr = loss(self.params, ValSamples, ValY)

      self.trainErrors.append(trainErr)
      self.valErrors.append(valErr)

      if(epoch % 10000 == 0):
        print('epoch: %d, error: %f' % (epoch, valErr))

      if((oldParams == self.params).all()):
        print('epoch: %d, error: %f' % (epoch, valErr))
        break
    
  def predict(self, samples: pd.DataFrame):
    Samples = samples.to_numpy()
    Samples = np.hstack((np.ones((Samples.shape[0], 1)), Samples))

    return h(self.params, Samples)

  def score(self, samples: pd.DataFrame, y: pd.DataFrame):
    Samples, Y = self.__dataSetup__(samples, y)

    preds = h(self.params, Samples)

    yMean = np.mean(Y)
    ssTot = np.sum((Y - yMean) ** 2)
    ssRes = np.sum((Y - preds) ** 2)
    
    return 1 - (ssRes / ssTot)
  

def StandardScaler(df: pd.DataFrame) -> pd.DataFrame:
    mean = df.mean()
    scale = df.std()

    return lambda x: (x - mean) / scale


def SplitDataset(df: pd.DataFrame, trainRatio=0.6, valRatio=0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  shuffled = df.sample(frac=1, random_state=69).reset_index(drop=True)

  totalSize = len(shuffled)
  trainSize = int(totalSize * trainRatio)
  valSize = int(totalSize * valRatio)
  testSize = totalSize - trainSize - valSize

  trainIndices = range(trainSize)
  valIndices = range(trainSize, trainSize + valSize)
  testIndices = range(trainSize + valSize, trainSize + valSize + testSize)

  train = shuffled.iloc[trainIndices]
  val = shuffled.iloc[valIndices]
  test = shuffled.iloc[testIndices]

  return (train, val, test)