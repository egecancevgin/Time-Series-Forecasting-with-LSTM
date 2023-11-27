# Time-Series-Forecasting-with-LSTM
Simple Weather Forecasting

İki dosya oluşturalım: forecast_functions.py ve main.py:
``` .sh
$ touch forecast_functions.py
$ touch main.py
```

'forecast_functions.py' dosyasının tepesinde modülleri indirelim:
``` python
import tensorflow as tf
import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
```

Aynı dosya içerisinde veri setimizi okuma fonksiyonu oluşturalım:
``` python
def read_file():
    """
    Reads and extracts the weather forecasting data
    :return: Dataframe
    """
    zipped_path = tf.keras.utils.get_file(
        origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
        fname='jena_climate_2009_2016.csv.zip',
        extract=True)
    csv_pth, _ = os.path.splittext(zipped_path)
    df = pd.read_csv(csv_pth)
    return csv_pth
 ```

Verimize bir göz atmak için main.py dosyamıza geçip main() fonksiyonu oluşturup çağıralım:
``` python
from forecast_functions import *


def main():
    my_csv = read_file()
    print(my_csv.head(), '\n\n')
    print(my_csv.info(), '\n\n')
    print(my_csv.describe(), '\n\n')


if __name__ == '__main__':
    main()
```

Çıktımız şu şekilde olacaktır:

![tsf_0](https://github.com/egecancevgin/Time-Series-Forecasting-with-LSTM/blob/main/TSF_1.png)

Fonksiyonlar dosyasına geri dönelim ve veri işlerken kullanacağımız train-test split işlemini gerçekleştirelim:
``` python
def df_to_X_y(df, window_size=5):
  """
  Convert the dataframe into training X and target y sets like this:
    [[[1], [2], [3], [4], [5]]] [6]
    [[[2], [3], [4], [5], [6]]] [7]
    [[[3], [4], [5], [6], [7]]] [8]
    :param df: Input dataframe
    :param window_size: Number of units in the training set
    :return: X and y
  """
  df_as_np = df.to_numpy()
  X = []
  y = []
  for i in range(len(df_as_np) - window_size):
    row = [[a] for a in df_as_np[i: i+window_size]]
    X.append(row)
    label = df_as_np[i + window_size]
    y.append(label)

  return np.array(X), np.array(y)
```

Bu şekilde verimiz örnek olarak 5 saat eğitilip bir saat etiket olarak kullanılacaktır.
