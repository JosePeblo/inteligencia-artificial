import numpy as np
import pandas as pd

def h(params: np.ndarray, samples: np.ndarray):
  '''
  Hypothesis function for a linear regression performs a matrix multiplication 
  with the params vector and the provided samples.
  IMPORTANT: this function asumes that the params have the bias on element
  
  Args:
    params (np.ndarray) coeficients for the linear regression as a 2d vector (n,1)
    samples (np.ndarray) matrix with rows of features stacked (m, n)
  
  Returns:
    np.ndarray: a column matrix (n,1) with the predictions of the lienear regression
  '''
  return samples @ params
  
def GD(params: np.ndarray, samples: np.ndarray, y: np.ndarray, alpha: float):
  '''
  Gradient descent for the linear model, computes the regression and
  return the updated coeficients based on the learning factor alpha
  
  Args:
  params (np.ndarray) coeficients for the linear regression as a 2d vector (n,1)
  samples (np.ndarray) matrix with rows of features stacked (m, n)
  y (np.ndarray) expected values for the regression (n, 1)
  alpha (float) learning rate

  Returns:
    np.ndarray: the adjusted coeficients (n,1)
  '''
  err = h(params, samples) - y
  gradient = (samples.T @ err) / samples.shape[0]
  return params - alpha * gradient

def loss(params: np.ndarray, samples: np.ndarray, y: np.ndarray):
  '''
  Computes the loss/cost function (MSE for the linear regression model

  Args:
  params (np.ndarray) coeficients for the linear regression as a 2d vector (n,1)
  samples (np.ndarray) matrix with rows of features stacked (m, n)
  y (np.ndarray) expected values for the regression (n, 1)

  Returns:
    float: MSE loss value
  '''
  hyp = h(params, samples)
  return np.square(hyp - y).mean()

class LinearRegression:
  def __init__(self):
    '''
    Initialize object internal variables
    '''
    self.params = None
    self.trainErrors = []
    self.valErrors = []

  def __dataSetup__(self, samples: pd.DataFrame, y: pd.DataFrame):
    '''
    Transform the pandas dataframes to numpy arrays and prepare the samples 
    with their corresponding column of ones for the bias

    Args:
      params (pd.DataFrame) coeficients for the linear regression as a 2d vector (n,1)
      y (pd.DataFrame) expected values for the regression (n, 1)

    Returns:
      tuple: samples and y as numpy arrays
    '''
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
    '''
    Train the linear regression model using gradient descent
    Args:
      trainSamples (pd.Dataframe) samples to train the model with
      trainY (pd.Dataframe) expected outcomes for the train set
      valSamples (pd.Dataframe) samples to validate the model with
      valY (pd.Dataframe) expected outcomes for the validation set
      alpha (float) learning rate
    '''
    
    # Reset errors on new fits
    self.trainErrors = []
    self.valErrors = []
    
    # Prepare the data
    TrainSamples, TrainY = self.__dataSetup__(trainSamples, trainY)
    ValSamples, ValY = self.__dataSetup__(valSamples, valY)

    # Initialize the parameters randomly
    self.params = np.random.random((TrainSamples.shape[1], 1))
    self.params[0][0] = 1 # The bias is still one lol

    for epoch in range(40000): # fixed epochs
      oldParams = self.params.copy()

      # Get the gradient
      self.params = GD(self.params, TrainSamples, TrainY, alpha)

      # Get the errors
      trainErr = loss(self.params, TrainSamples, TrainY)
      valErr = loss(self.params, ValSamples, ValY)

      self.trainErrors.append(trainErr)
      self.valErrors.append(valErr)

      if(epoch % 10000 == 0): # Print every 10000 epochs to prevent delays
        print('epoch: %d, error: %f' % (epoch, valErr))

      if((oldParams == self.params).all()):
        print('epoch: %d, error: %f' % (epoch, valErr))
        break
    
  def predict(self, samples: pd.DataFrame):
    '''
    Generate the predictions for a set of data
    Args:
      samples (pd.Dataframe) the samples to make the predictions
    Returns:
      np.ndarray: A vector with the result of every prediction
    '''
    Samples = samples.to_numpy()
    Samples = np.hstack((np.ones((Samples.shape[0], 1)), Samples))

    return h(self.params, Samples)

  def score(self, samples: pd.DataFrame, y: pd.DataFrame):
    '''
    Calculate the R2 for the model 
    Args:
      samples (pd.Dataframe) samples to do the predictions
      y (pd.Dataframe) expected outcomes of the regression
    Returns:
      float: R2 value for the prediction
    '''
    Samples, Y = self.__dataSetup__(samples, y)

    preds = h(self.params, Samples)

    yMean = np.mean(Y)
    ssTot = np.sum((Y - yMean) ** 2)
    ssRes = np.sum((Y - preds) ** 2)
    
    return 1 - (ssRes / ssTot)
  

def StandardScaler(df: pd.DataFrame):
  '''
  Creates a scaler function that performs z-scaling on a dataframe
  z-scaling aims to turn the distribution to something closer to a normal distribution
  with mean 0 and deviation 1

  Args:
    df (pd.Dataframe) the set of values that will be used to create the scaler
  
  Returns:
    lambda: function to perform the scaling after
  '''
  mean = df.mean()
  scale = df.std()

  return lambda x: (x - mean) / scale


def SplitDataset(df: pd.DataFrame, trainRatio=0.6, valRatio=0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  '''
  Split the original dataset into three datasets: train, validation, and test
  
  IMPORTANT: The remainder of both ratios will be the test set

  Args:
    df (pd.DataFrame) dataset to split
    trainRatio (float) the ratio that the training data will take
    valRatio (float) the ratio that the validation set will take

  Returns:
    tuple: the separated datasets

  '''
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