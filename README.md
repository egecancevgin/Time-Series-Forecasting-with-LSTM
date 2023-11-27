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

