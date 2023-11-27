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

Şimdi veri işleme aşamasına geçelim, bu aşamada önce verimizi ufaltalım, her saat başı bir satırımız olsun. Normalde her 10 dakikada bir verimiz vardı ancak bu çok fazla.

Bir datetime sütunu oluşturalım, ismi 'Date Time' olsun, hedef sütun olarak da 'T (degC)' sütununu seçelim, minik bir plot da oluşturalım, işlem bitince incelemek için.

Deminki df_to_X_y() fonksiyonunu çağırıp içine hedef sütunu koyalım, çıktıyı da X1-y1 olarak ayarlayalım.
Bundan sonra da eğitim-validasyon-test kümelerine ayrım yapalım, %85-%7-%7 gibi bir ayrım olacaktır.

Sonra bu parçaları döndürelim ve bir görüntüleme olsun diye print de ettirelim:
``` python
def preprocessing(df):
  """
  Preprocesses the dataframe
  :return: Processed dataframe, train-validation-test sets
  """
  # Triming the data by choosing only once an hour
  df = df[5::6]

  # Creating a datetime column by using the index
  df.index = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')

  # Temp is the target column, let's take a look at it
  temp = df['T (degC)']
  temp.plot()

  # Splitting the time series to training and the target label
  WINDOW_SIZE = 5
  X1, y1 = df_to_X_y(temp, WINDOW_SIZE)

  # Splitting into training-validation-test sets %85-%7-%7
  X_train1, y_train1 = X1[:60000], y1[:60000]
  X_val1, y_val1 = X1[60000:65000], y1[60000:65000]
  X_test1, y_test1 = X1[65000:], y1[65000:]

  # Let's take a look at the splits
  print(
      X_train1.shape, y_train1.shape, X_val1.shape,
      y_val1.shape, X_test1.shape, y_test1.shape
  )

  return X_train1, y_train1, X_val1, y_val1, X_test1, y_test1
```

Kontrol etmek için main.py dosyamızdaki main() fonksiyonunun içinde çağıralım ve inceleyelim:
``` python
def main():
    my_df = read_file()
    #print(my_csv.head(), '\n\n')
    #print(my_csv.info(), '\n\n')
    #print(my_csv.describe(), '\n\n')
    Xt1, yt1, Xv1, yv1, Xts1, yts1 = preprocessing(my_df)


if __name__ == '__main__':
    main()
```

