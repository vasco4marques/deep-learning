import numpy as np

def relu(x):   
      return np.maximum(0, x)

arr = np.array([np.inf,np.inf,np.inf,np.inf])
arr = np.expand_dims(arr, axis=1)

arrMax= relu(arr)

print(arrMax)