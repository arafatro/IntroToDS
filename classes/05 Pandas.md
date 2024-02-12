# Pandas

## Import


```python
import warnings
warnings.simplefilter(action='ignore')

import pandas as pd
import numpy as np
```

## Series


```python
s = pd.Series()
s
```




    Series([], dtype: float64)




```python
myList = [10, 20, 30, 40, 50]
myArray = np.array(myList)

myArray
```




    array([10, 20, 30, 40, 50])




```python
pd.Series(data = myArray)
```




    0    10
    1    20
    2    30
    3    40
    4    50
    dtype: int64




```python
labels = ['a', 'b', 'c', 'd', 'e']
pd.Series(myArray, labels)
```




    a    10
    b    20
    c    30
    d    40
    e    50
    dtype: int64



## Lists


```python
pd.Series(myList)
```




    0    10
    1    20
    2    30
    3    40
    4    50
    dtype: int64




```python
pd.Series(myList, labels)
```




    a    10
    b    20
    c    30
    d    40
    e    50
    dtype: int64



## Dictionaries


```python
myDict = {
    "name": "Jose",
    "position": "PhD student",
    "year": 2019
}
pd.Series(myDict)
```




    name               Jose
    position    PhD student
    year               2019
    dtype: object



### Data in Series


```python
pd.Series(data = labels)
```




    0    a
    1    b
    2    c
    3    d
    4    e
    dtype: object



## Accessing Data from Series


```python
s1 = pd.Series(myDict)

s1[1]
```




    'PhD student'




```python
data = [1, 2, 3, 4, 5]
indexes = ['a', 'b', 'c', 'd', 'e']

s = pd.Series(data, indexes)

print(s[['a', 'd', 'e']])
```

    a    1
    d    4
    e    5
    dtype: int64


## Data Frames


```python
ids = {
    0:{
        "name": "Jose",
        "position": "PhD student",
        "year": 2019
    },
    1:{
        "name": "Ana",
        "position": "Lawyer",
        "year": 2020
    },
    2:{
        "name": "Vanda",
        "position": "Nurse",
        "year": 1990
    }
}

ids
```




    {0: {'name': 'Jose', 'position': 'PhD student', 'year': 2019},
     1: {'name': 'Ana', 'position': 'Lawyer', 'year': 2020},
     2: {'name': 'Vanda', 'position': 'Nurse', 'year': 1990}}




```python
df = pd.DataFrame.from_dict(ids, orient='index')
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>position</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Jose</td>
      <td>PhD student</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ana</td>
      <td>Lawyer</td>
      <td>2020</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Vanda</td>
      <td>Nurse</td>
      <td>1990</td>
    </tr>
  </tbody>
</table>
</div>



### From Matrix


```python
from numpy.random import randn
np.random.seed(101)

df = pd.DataFrame(randn(10, 6), columns="A B C D E F".split())
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.706850</td>
      <td>0.628133</td>
      <td>0.907969</td>
      <td>0.503826</td>
      <td>0.651118</td>
      <td>-0.319318</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.848077</td>
      <td>0.605965</td>
      <td>-2.018168</td>
      <td>0.740122</td>
      <td>0.528813</td>
      <td>-0.589001</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.188695</td>
      <td>-0.758872</td>
      <td>-0.933237</td>
      <td>0.955057</td>
      <td>0.190794</td>
      <td>1.978757</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.605967</td>
      <td>0.683509</td>
      <td>0.302665</td>
      <td>1.693723</td>
      <td>-1.706086</td>
      <td>-1.159119</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.134841</td>
      <td>0.390528</td>
      <td>0.166905</td>
      <td>0.184502</td>
      <td>0.807706</td>
      <td>0.072960</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.638787</td>
      <td>0.329646</td>
      <td>-0.497104</td>
      <td>-0.754070</td>
      <td>-0.943406</td>
      <td>0.484752</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.116773</td>
      <td>1.901755</td>
      <td>0.238127</td>
      <td>1.996652</td>
      <td>-0.993263</td>
      <td>0.196800</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-1.136645</td>
      <td>0.000366</td>
      <td>1.025984</td>
      <td>-0.156598</td>
      <td>-0.031579</td>
      <td>0.649826</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2.154846</td>
      <td>-0.610259</td>
      <td>-0.755325</td>
      <td>-0.346419</td>
      <td>0.147027</td>
      <td>-0.479448</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.558769</td>
      <td>1.024810</td>
      <td>-0.925874</td>
      <td>1.862864</td>
      <td>-1.133817</td>
      <td>0.610478</td>
    </tr>
  </tbody>
</table>
</div>



### Overview


```python
df.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.661758</td>
      <td>0.419558</td>
      <td>-0.248806</td>
      <td>0.667966</td>
      <td>-0.248269</td>
      <td>0.144669</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.381670</td>
      <td>0.770644</td>
      <td>0.945926</td>
      <td>0.961754</td>
      <td>0.873784</td>
      <td>0.869387</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.136645</td>
      <td>-0.758872</td>
      <td>-2.018168</td>
      <td>-0.754070</td>
      <td>-1.706086</td>
      <td>-1.159119</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.130324</td>
      <td>0.082686</td>
      <td>-0.883237</td>
      <td>-0.071323</td>
      <td>-0.980799</td>
      <td>-0.439416</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.373732</td>
      <td>0.498247</td>
      <td>-0.165100</td>
      <td>0.621974</td>
      <td>0.057724</td>
      <td>0.134880</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.775832</td>
      <td>0.669665</td>
      <td>0.286531</td>
      <td>1.509056</td>
      <td>0.444309</td>
      <td>0.579046</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.706850</td>
      <td>1.901755</td>
      <td>1.025984</td>
      <td>1.996652</td>
      <td>0.807706</td>
      <td>1.978757</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.D.value_counts()
```




     1.693723    1
     1.862864    1
     0.955057    1
    -0.346419    1
     1.996652    1
    -0.754070    1
     0.503826    1
     0.184502    1
     0.740122    1
    -0.156598    1
    Name: D, dtype: int64




```python
df.dtypes.value_counts()
```




    float64    6
    dtype: int64



### Indexing


```python
df['D']
```




    0    0.503826
    1    0.740122
    2    0.955057
    3    1.693723
    4    0.184502
    5   -0.754070
    6    1.996652
    7   -0.156598
    8   -0.346419
    9    1.862864
    Name: D, dtype: float64




```python
df[['D', 'A']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>D</th>
      <th>A</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.503826</td>
      <td>2.706850</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.740122</td>
      <td>-0.848077</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.955057</td>
      <td>0.188695</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.693723</td>
      <td>2.605967</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.184502</td>
      <td>-0.134841</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.754070</td>
      <td>0.638787</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.996652</td>
      <td>-0.116773</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-0.156598</td>
      <td>-1.136645</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-0.346419</td>
      <td>2.154846</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.862864</td>
      <td>0.558769</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Not recommend
df.D
```




    0    0.503826
    1    0.740122
    2    0.955057
    3    1.693723
    4    0.184502
    5   -0.754070
    6    1.996652
    7   -0.156598
    8   -0.346419
    9    1.862864
    Name: D, dtype: float64




```python
type(df['A'])
```




    pandas.core.series.Series



## Creating and Removing


```python
df['G'] = df['B'] + df['D']
df['H'] = df['A'] - df['C']

df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
      <th>G</th>
      <th>H</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.706850</td>
      <td>0.628133</td>
      <td>0.907969</td>
      <td>0.503826</td>
      <td>0.651118</td>
      <td>-0.319318</td>
      <td>1.131958</td>
      <td>1.798880</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.848077</td>
      <td>0.605965</td>
      <td>-2.018168</td>
      <td>0.740122</td>
      <td>0.528813</td>
      <td>-0.589001</td>
      <td>1.346087</td>
      <td>1.170091</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.188695</td>
      <td>-0.758872</td>
      <td>-0.933237</td>
      <td>0.955057</td>
      <td>0.190794</td>
      <td>1.978757</td>
      <td>0.196184</td>
      <td>1.121933</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.605967</td>
      <td>0.683509</td>
      <td>0.302665</td>
      <td>1.693723</td>
      <td>-1.706086</td>
      <td>-1.159119</td>
      <td>2.377232</td>
      <td>2.303302</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.134841</td>
      <td>0.390528</td>
      <td>0.166905</td>
      <td>0.184502</td>
      <td>0.807706</td>
      <td>0.072960</td>
      <td>0.575030</td>
      <td>-0.301745</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.638787</td>
      <td>0.329646</td>
      <td>-0.497104</td>
      <td>-0.754070</td>
      <td>-0.943406</td>
      <td>0.484752</td>
      <td>-0.424423</td>
      <td>1.135891</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.116773</td>
      <td>1.901755</td>
      <td>0.238127</td>
      <td>1.996652</td>
      <td>-0.993263</td>
      <td>0.196800</td>
      <td>3.898407</td>
      <td>-0.354900</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-1.136645</td>
      <td>0.000366</td>
      <td>1.025984</td>
      <td>-0.156598</td>
      <td>-0.031579</td>
      <td>0.649826</td>
      <td>-0.156231</td>
      <td>-2.162629</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2.154846</td>
      <td>-0.610259</td>
      <td>-0.755325</td>
      <td>-0.346419</td>
      <td>0.147027</td>
      <td>-0.479448</td>
      <td>-0.956677</td>
      <td>2.910172</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.558769</td>
      <td>1.024810</td>
      <td>-0.925874</td>
      <td>1.862864</td>
      <td>-1.133817</td>
      <td>0.610478</td>
      <td>2.887674</td>
      <td>1.484644</td>
    </tr>
  </tbody>
</table>
</div>




```python
#df2 = df.drop('G', axis = 1)
df.drop('G', axis = 1, inplace=True)
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
      <th>H</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.706850</td>
      <td>0.628133</td>
      <td>0.907969</td>
      <td>0.503826</td>
      <td>0.651118</td>
      <td>-0.319318</td>
      <td>1.798880</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.848077</td>
      <td>0.605965</td>
      <td>-2.018168</td>
      <td>0.740122</td>
      <td>0.528813</td>
      <td>-0.589001</td>
      <td>1.170091</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.188695</td>
      <td>-0.758872</td>
      <td>-0.933237</td>
      <td>0.955057</td>
      <td>0.190794</td>
      <td>1.978757</td>
      <td>1.121933</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.605967</td>
      <td>0.683509</td>
      <td>0.302665</td>
      <td>1.693723</td>
      <td>-1.706086</td>
      <td>-1.159119</td>
      <td>2.303302</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.134841</td>
      <td>0.390528</td>
      <td>0.166905</td>
      <td>0.184502</td>
      <td>0.807706</td>
      <td>0.072960</td>
      <td>-0.301745</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.638787</td>
      <td>0.329646</td>
      <td>-0.497104</td>
      <td>-0.754070</td>
      <td>-0.943406</td>
      <td>0.484752</td>
      <td>1.135891</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.116773</td>
      <td>1.901755</td>
      <td>0.238127</td>
      <td>1.996652</td>
      <td>-0.993263</td>
      <td>0.196800</td>
      <td>-0.354900</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-1.136645</td>
      <td>0.000366</td>
      <td>1.025984</td>
      <td>-0.156598</td>
      <td>-0.031579</td>
      <td>0.649826</td>
      <td>-2.162629</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2.154846</td>
      <td>-0.610259</td>
      <td>-0.755325</td>
      <td>-0.346419</td>
      <td>0.147027</td>
      <td>-0.479448</td>
      <td>2.910172</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.558769</td>
      <td>1.024810</td>
      <td>-0.925874</td>
      <td>1.862864</td>
      <td>-1.133817</td>
      <td>0.610478</td>
      <td>1.484644</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.drop(3, axis = 0, inplace=True)
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
      <th>H</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.706850</td>
      <td>0.628133</td>
      <td>0.907969</td>
      <td>0.503826</td>
      <td>0.651118</td>
      <td>-0.319318</td>
      <td>1.798880</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.848077</td>
      <td>0.605965</td>
      <td>-2.018168</td>
      <td>0.740122</td>
      <td>0.528813</td>
      <td>-0.589001</td>
      <td>1.170091</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.188695</td>
      <td>-0.758872</td>
      <td>-0.933237</td>
      <td>0.955057</td>
      <td>0.190794</td>
      <td>1.978757</td>
      <td>1.121933</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.134841</td>
      <td>0.390528</td>
      <td>0.166905</td>
      <td>0.184502</td>
      <td>0.807706</td>
      <td>0.072960</td>
      <td>-0.301745</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.638787</td>
      <td>0.329646</td>
      <td>-0.497104</td>
      <td>-0.754070</td>
      <td>-0.943406</td>
      <td>0.484752</td>
      <td>1.135891</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.116773</td>
      <td>1.901755</td>
      <td>0.238127</td>
      <td>1.996652</td>
      <td>-0.993263</td>
      <td>0.196800</td>
      <td>-0.354900</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-1.136645</td>
      <td>0.000366</td>
      <td>1.025984</td>
      <td>-0.156598</td>
      <td>-0.031579</td>
      <td>0.649826</td>
      <td>-2.162629</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2.154846</td>
      <td>-0.610259</td>
      <td>-0.755325</td>
      <td>-0.346419</td>
      <td>0.147027</td>
      <td>-0.479448</td>
      <td>2.910172</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.558769</td>
      <td>1.024810</td>
      <td>-0.925874</td>
      <td>1.862864</td>
      <td>-1.133817</td>
      <td>0.610478</td>
      <td>1.484644</td>
    </tr>
  </tbody>
</table>
</div>



## Selecting


```python
df.iloc[4]
```




    A    0.638787
    B    0.329646
    C   -0.497104
    D   -0.754070
    E   -0.943406
    F    0.484752
    H    1.135891
    Name: 5, dtype: float64




```python
df.iloc[1, 4]
```




    0.5288134940893595




```python
df.iloc[:3, 2]
```




    0    0.907969
    1   -2.018168
    2   -0.933237
    Name: C, dtype: float64




```python
df.loc[1]
```




    A   -0.848077
    B    0.605965
    C   -2.018168
    D    0.740122
    E    0.528813
    F   -0.589001
    H    1.170091
    Name: 1, dtype: float64



### Selecting a subset


```python
df.loc[:3,'A':'C']
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.706850</td>
      <td>0.628133</td>
      <td>0.907969</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.848077</td>
      <td>0.605965</td>
      <td>-2.018168</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.188695</td>
      <td>-0.758872</td>
      <td>-0.933237</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc[1,'E']
```




    0.5288134940893595




```python
df.loc[[1,5], ['A', 'C']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>-0.848077</td>
      <td>-2.018168</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.638787</td>
      <td>-0.497104</td>
    </tr>
  </tbody>
</table>
</div>



## Conditional Selection


```python
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
      <th>H</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.706850</td>
      <td>0.628133</td>
      <td>0.907969</td>
      <td>0.503826</td>
      <td>0.651118</td>
      <td>-0.319318</td>
      <td>1.798880</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.848077</td>
      <td>0.605965</td>
      <td>-2.018168</td>
      <td>0.740122</td>
      <td>0.528813</td>
      <td>-0.589001</td>
      <td>1.170091</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.188695</td>
      <td>-0.758872</td>
      <td>-0.933237</td>
      <td>0.955057</td>
      <td>0.190794</td>
      <td>1.978757</td>
      <td>1.121933</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.134841</td>
      <td>0.390528</td>
      <td>0.166905</td>
      <td>0.184502</td>
      <td>0.807706</td>
      <td>0.072960</td>
      <td>-0.301745</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.638787</td>
      <td>0.329646</td>
      <td>-0.497104</td>
      <td>-0.754070</td>
      <td>-0.943406</td>
      <td>0.484752</td>
      <td>1.135891</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.116773</td>
      <td>1.901755</td>
      <td>0.238127</td>
      <td>1.996652</td>
      <td>-0.993263</td>
      <td>0.196800</td>
      <td>-0.354900</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-1.136645</td>
      <td>0.000366</td>
      <td>1.025984</td>
      <td>-0.156598</td>
      <td>-0.031579</td>
      <td>0.649826</td>
      <td>-2.162629</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2.154846</td>
      <td>-0.610259</td>
      <td>-0.755325</td>
      <td>-0.346419</td>
      <td>0.147027</td>
      <td>-0.479448</td>
      <td>2.910172</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.558769</td>
      <td>1.024810</td>
      <td>-0.925874</td>
      <td>1.862864</td>
      <td>-1.133817</td>
      <td>0.610478</td>
      <td>1.484644</td>
    </tr>
  </tbody>
</table>
</div>




```python
df > 0
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
      <th>H</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6</th>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>9</th>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df > 0] #NaN = Not a Number
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
      <th>H</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.706850</td>
      <td>0.628133</td>
      <td>0.907969</td>
      <td>0.503826</td>
      <td>0.651118</td>
      <td>NaN</td>
      <td>1.798880</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>0.605965</td>
      <td>NaN</td>
      <td>0.740122</td>
      <td>0.528813</td>
      <td>NaN</td>
      <td>1.170091</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.188695</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.955057</td>
      <td>0.190794</td>
      <td>1.978757</td>
      <td>1.121933</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>0.390528</td>
      <td>0.166905</td>
      <td>0.184502</td>
      <td>0.807706</td>
      <td>0.072960</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.638787</td>
      <td>0.329646</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.484752</td>
      <td>1.135891</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NaN</td>
      <td>1.901755</td>
      <td>0.238127</td>
      <td>1.996652</td>
      <td>NaN</td>
      <td>0.196800</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NaN</td>
      <td>0.000366</td>
      <td>1.025984</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.649826</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2.154846</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.147027</td>
      <td>NaN</td>
      <td>2.910172</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.558769</td>
      <td>1.024810</td>
      <td>NaN</td>
      <td>1.862864</td>
      <td>NaN</td>
      <td>0.610478</td>
      <td>1.484644</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df['E']>0]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
      <th>H</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.706850</td>
      <td>0.628133</td>
      <td>0.907969</td>
      <td>0.503826</td>
      <td>0.651118</td>
      <td>-0.319318</td>
      <td>1.798880</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.848077</td>
      <td>0.605965</td>
      <td>-2.018168</td>
      <td>0.740122</td>
      <td>0.528813</td>
      <td>-0.589001</td>
      <td>1.170091</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.188695</td>
      <td>-0.758872</td>
      <td>-0.933237</td>
      <td>0.955057</td>
      <td>0.190794</td>
      <td>1.978757</td>
      <td>1.121933</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.134841</td>
      <td>0.390528</td>
      <td>0.166905</td>
      <td>0.184502</td>
      <td>0.807706</td>
      <td>0.072960</td>
      <td>-0.301745</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2.154846</td>
      <td>-0.610259</td>
      <td>-0.755325</td>
      <td>-0.346419</td>
      <td>0.147027</td>
      <td>-0.479448</td>
      <td>2.910172</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df['D']>0][['A', 'F']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.706850</td>
      <td>-0.319318</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.848077</td>
      <td>-0.589001</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.188695</td>
      <td>1.978757</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.134841</td>
      <td>0.072960</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.116773</td>
      <td>0.196800</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.558769</td>
      <td>0.610478</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[(df['A']>0)&(df['C']<0)]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
      <th>H</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>0.188695</td>
      <td>-0.758872</td>
      <td>-0.933237</td>
      <td>0.955057</td>
      <td>0.190794</td>
      <td>1.978757</td>
      <td>1.121933</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.638787</td>
      <td>0.329646</td>
      <td>-0.497104</td>
      <td>-0.754070</td>
      <td>-0.943406</td>
      <td>0.484752</td>
      <td>1.135891</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2.154846</td>
      <td>-0.610259</td>
      <td>-0.755325</td>
      <td>-0.346419</td>
      <td>0.147027</td>
      <td>-0.479448</td>
      <td>2.910172</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.558769</td>
      <td>1.024810</td>
      <td>-0.925874</td>
      <td>1.862864</td>
      <td>-1.133817</td>
      <td>0.610478</td>
      <td>1.484644</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[(df['A']>0) & (df['C']<0)]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
      <th>H</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>0.188695</td>
      <td>-0.758872</td>
      <td>-0.933237</td>
      <td>0.955057</td>
      <td>0.190794</td>
      <td>1.978757</td>
      <td>1.121933</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.638787</td>
      <td>0.329646</td>
      <td>-0.497104</td>
      <td>-0.754070</td>
      <td>-0.943406</td>
      <td>0.484752</td>
      <td>1.135891</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2.154846</td>
      <td>-0.610259</td>
      <td>-0.755325</td>
      <td>-0.346419</td>
      <td>0.147027</td>
      <td>-0.479448</td>
      <td>2.910172</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.558769</td>
      <td>1.024810</td>
      <td>-0.925874</td>
      <td>1.862864</td>
      <td>-1.133817</td>
      <td>0.610478</td>
      <td>1.484644</td>
    </tr>
  </tbody>
</table>
</div>



## Resetting


```python
df = df.loc[:5, :]
df.reset_index()
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
      <th>H</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.706850</td>
      <td>0.628133</td>
      <td>0.907969</td>
      <td>0.503826</td>
      <td>0.651118</td>
      <td>-0.319318</td>
      <td>1.798880</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.848077</td>
      <td>0.605965</td>
      <td>-2.018168</td>
      <td>0.740122</td>
      <td>0.528813</td>
      <td>-0.589001</td>
      <td>1.170091</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.188695</td>
      <td>-0.758872</td>
      <td>-0.933237</td>
      <td>0.955057</td>
      <td>0.190794</td>
      <td>1.978757</td>
      <td>1.121933</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.134841</td>
      <td>0.390528</td>
      <td>0.166905</td>
      <td>0.184502</td>
      <td>0.807706</td>
      <td>0.072960</td>
      <td>-0.301745</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.638787</td>
      <td>0.329646</td>
      <td>-0.497104</td>
      <td>-0.754070</td>
      <td>-0.943406</td>
      <td>0.484752</td>
      <td>1.135891</td>
    </tr>
  </tbody>
</table>
</div>




```python
newIndex = "IN1 IN2 IN3 IN4 IN5".split()
df["newInd"] = newIndex
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
      <th>H</th>
      <th>newInd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.706850</td>
      <td>0.628133</td>
      <td>0.907969</td>
      <td>0.503826</td>
      <td>0.651118</td>
      <td>-0.319318</td>
      <td>1.798880</td>
      <td>IN1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.848077</td>
      <td>0.605965</td>
      <td>-2.018168</td>
      <td>0.740122</td>
      <td>0.528813</td>
      <td>-0.589001</td>
      <td>1.170091</td>
      <td>IN2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.188695</td>
      <td>-0.758872</td>
      <td>-0.933237</td>
      <td>0.955057</td>
      <td>0.190794</td>
      <td>1.978757</td>
      <td>1.121933</td>
      <td>IN3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.134841</td>
      <td>0.390528</td>
      <td>0.166905</td>
      <td>0.184502</td>
      <td>0.807706</td>
      <td>0.072960</td>
      <td>-0.301745</td>
      <td>IN4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.638787</td>
      <td>0.329646</td>
      <td>-0.497104</td>
      <td>-0.754070</td>
      <td>-0.943406</td>
      <td>0.484752</td>
      <td>1.135891</td>
      <td>IN5</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.set_index('newInd')
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
      <th>H</th>
    </tr>
    <tr>
      <th>newInd</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>IN1</th>
      <td>2.706850</td>
      <td>0.628133</td>
      <td>0.907969</td>
      <td>0.503826</td>
      <td>0.651118</td>
      <td>-0.319318</td>
      <td>1.798880</td>
    </tr>
    <tr>
      <th>IN2</th>
      <td>-0.848077</td>
      <td>0.605965</td>
      <td>-2.018168</td>
      <td>0.740122</td>
      <td>0.528813</td>
      <td>-0.589001</td>
      <td>1.170091</td>
    </tr>
    <tr>
      <th>IN3</th>
      <td>0.188695</td>
      <td>-0.758872</td>
      <td>-0.933237</td>
      <td>0.955057</td>
      <td>0.190794</td>
      <td>1.978757</td>
      <td>1.121933</td>
    </tr>
    <tr>
      <th>IN4</th>
      <td>-0.134841</td>
      <td>0.390528</td>
      <td>0.166905</td>
      <td>0.184502</td>
      <td>0.807706</td>
      <td>0.072960</td>
      <td>-0.301745</td>
    </tr>
    <tr>
      <th>IN5</th>
      <td>0.638787</td>
      <td>0.329646</td>
      <td>-0.497104</td>
      <td>-0.754070</td>
      <td>-0.943406</td>
      <td>0.484752</td>
      <td>1.135891</td>
    </tr>
  </tbody>
</table>
</div>




```python
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
      <th>H</th>
      <th>newInd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.706850</td>
      <td>0.628133</td>
      <td>0.907969</td>
      <td>0.503826</td>
      <td>0.651118</td>
      <td>-0.319318</td>
      <td>1.798880</td>
      <td>IN1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.848077</td>
      <td>0.605965</td>
      <td>-2.018168</td>
      <td>0.740122</td>
      <td>0.528813</td>
      <td>-0.589001</td>
      <td>1.170091</td>
      <td>IN2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.188695</td>
      <td>-0.758872</td>
      <td>-0.933237</td>
      <td>0.955057</td>
      <td>0.190794</td>
      <td>1.978757</td>
      <td>1.121933</td>
      <td>IN3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.134841</td>
      <td>0.390528</td>
      <td>0.166905</td>
      <td>0.184502</td>
      <td>0.807706</td>
      <td>0.072960</td>
      <td>-0.301745</td>
      <td>IN4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.638787</td>
      <td>0.329646</td>
      <td>-0.497104</td>
      <td>-0.754070</td>
      <td>-0.943406</td>
      <td>0.484752</td>
      <td>1.135891</td>
      <td>IN5</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.set_index('newInd', inplace=True)
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
      <th>H</th>
    </tr>
    <tr>
      <th>newInd</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>IN1</th>
      <td>2.706850</td>
      <td>0.628133</td>
      <td>0.907969</td>
      <td>0.503826</td>
      <td>0.651118</td>
      <td>-0.319318</td>
      <td>1.798880</td>
    </tr>
    <tr>
      <th>IN2</th>
      <td>-0.848077</td>
      <td>0.605965</td>
      <td>-2.018168</td>
      <td>0.740122</td>
      <td>0.528813</td>
      <td>-0.589001</td>
      <td>1.170091</td>
    </tr>
    <tr>
      <th>IN3</th>
      <td>0.188695</td>
      <td>-0.758872</td>
      <td>-0.933237</td>
      <td>0.955057</td>
      <td>0.190794</td>
      <td>1.978757</td>
      <td>1.121933</td>
    </tr>
    <tr>
      <th>IN4</th>
      <td>-0.134841</td>
      <td>0.390528</td>
      <td>0.166905</td>
      <td>0.184502</td>
      <td>0.807706</td>
      <td>0.072960</td>
      <td>-0.301745</td>
    </tr>
    <tr>
      <th>IN5</th>
      <td>0.638787</td>
      <td>0.329646</td>
      <td>-0.497104</td>
      <td>-0.754070</td>
      <td>-0.943406</td>
      <td>0.484752</td>
      <td>1.135891</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.reset_index()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>newInd</th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
      <th>H</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>IN1</td>
      <td>2.706850</td>
      <td>0.628133</td>
      <td>0.907969</td>
      <td>0.503826</td>
      <td>0.651118</td>
      <td>-0.319318</td>
      <td>1.798880</td>
    </tr>
    <tr>
      <th>1</th>
      <td>IN2</td>
      <td>-0.848077</td>
      <td>0.605965</td>
      <td>-2.018168</td>
      <td>0.740122</td>
      <td>0.528813</td>
      <td>-0.589001</td>
      <td>1.170091</td>
    </tr>
    <tr>
      <th>2</th>
      <td>IN3</td>
      <td>0.188695</td>
      <td>-0.758872</td>
      <td>-0.933237</td>
      <td>0.955057</td>
      <td>0.190794</td>
      <td>1.978757</td>
      <td>1.121933</td>
    </tr>
    <tr>
      <th>3</th>
      <td>IN4</td>
      <td>-0.134841</td>
      <td>0.390528</td>
      <td>0.166905</td>
      <td>0.184502</td>
      <td>0.807706</td>
      <td>0.072960</td>
      <td>-0.301745</td>
    </tr>
    <tr>
      <th>4</th>
      <td>IN5</td>
      <td>0.638787</td>
      <td>0.329646</td>
      <td>-0.497104</td>
      <td>-0.754070</td>
      <td>-0.943406</td>
      <td>0.484752</td>
      <td>1.135891</td>
    </tr>
  </tbody>
</table>
</div>



## Multi-index and Hierarchy


```python
outside = ['G1', 'G1', 'G1', 'G2', 'G2', 'G2']
inside = [1,2,3, 1,2,3]

hIndex = list(zip(outside, inside))
hIndex = pd.MultiIndex.from_tuples(hIndex)

hIndex
```




    MultiIndex([('G1', 1),
                ('G1', 2),
                ('G1', 3),
                ('G2', 1),
                ('G2', 2),
                ('G2', 3)],
               )




```python
df = pd.DataFrame(randn(6, 2), index = hIndex, columns="A B".split())
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">G1</th>
      <th>1</th>
      <td>0.386030</td>
      <td>2.084019</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.376519</td>
      <td>0.230336</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.681209</td>
      <td>1.035125</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">G2</th>
      <th>1</th>
      <td>-0.031160</td>
      <td>1.939932</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.005187</td>
      <td>-0.741790</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.187125</td>
      <td>-0.732845</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc['G2']
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>-0.031160</td>
      <td>1.939932</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.005187</td>
      <td>-0.741790</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.187125</td>
      <td>-0.732845</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc['G1'].loc[1]
```




    A    0.386030
    B    2.084019
    Name: 1, dtype: float64




```python
df.index.names
```




    FrozenList([None, None])




```python
df.index.names = ['Group', 'ID']
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>A</th>
      <th>B</th>
    </tr>
    <tr>
      <th>Group</th>
      <th>ID</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">G1</th>
      <th>1</th>
      <td>0.386030</td>
      <td>2.084019</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.376519</td>
      <td>0.230336</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.681209</td>
      <td>1.035125</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">G2</th>
      <th>1</th>
      <td>-0.031160</td>
      <td>1.939932</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.005187</td>
      <td>-0.741790</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.187125</td>
      <td>-0.732845</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.xs('G1')
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
    </tr>
    <tr>
      <th>ID</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.386030</td>
      <td>2.084019</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.376519</td>
      <td>0.230336</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.681209</td>
      <td>1.035125</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.xs(['G2', 2])
```




    A   -1.005187
    B   -0.741790
    Name: (G2, 2), dtype: float64




```python
df.xs(1, level='ID')
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
    </tr>
    <tr>
      <th>Group</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>G1</th>
      <td>0.38603</td>
      <td>2.084019</td>
    </tr>
    <tr>
      <th>G2</th>
      <td>-0.03116</td>
      <td>1.939932</td>
    </tr>
  </tbody>
</table>
</div>



## Missing values


```python
df = pd.DataFrame({'A':[1, 2, np.nan],
                   'B':[4, np.nan, np.nan],
                   'C':[7.0,8,9]
                  })
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>4.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>NaN</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>9.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.dropna()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>4.0</td>
      <td>7.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.dropna(axis=1)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.dropna(thresh=2)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>4.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>NaN</td>
      <td>8.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.fillna(value = -1)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>4.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>-1.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>9.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['A'].fillna(value = df['A'].mean(), inplace=True)
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>4.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>NaN</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.5</td>
      <td>NaN</td>
      <td>9.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['B'].fillna(value = df['B'].mean(), inplace=True)
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>4.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>4.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.5</td>
      <td>4.0</td>
      <td>9.0</td>
    </tr>
  </tbody>
</table>
</div>



## Group by


```python
data = {'Company':['Google','Google', 'Apple', 'Apple', 'Facebook','Facebook'],
        'Person':['Sam','Charlie','Amy','Sally','Carl','Sarah'],
        'Sales':[200, 150, 300, 100, 250, 350]
       }

df = pd.DataFrame(data)
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Company</th>
      <th>Person</th>
      <th>Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Google</td>
      <td>Sam</td>
      <td>200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Google</td>
      <td>Charlie</td>
      <td>150</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Apple</td>
      <td>Amy</td>
      <td>300</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Apple</td>
      <td>Sally</td>
      <td>100</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Facebook</td>
      <td>Carl</td>
      <td>250</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Facebook</td>
      <td>Sarah</td>
      <td>350</td>
    </tr>
  </tbody>
</table>
</div>




```python
dfGroup = df.groupby("Company")
dfGroup
```




    <pandas.core.groupby.generic.DataFrameGroupBy object at 0x7fb24e65b250>




```python
for name, group in dfGroup:
    print(name)
    print(group)
    print()
```

    Apple
      Company Person  Sales
    2   Apple    Amy    300
    3   Apple  Sally    100
    
    Facebook
        Company Person  Sales
    4  Facebook   Carl    250
    5  Facebook  Sarah    350
    
    Google
      Company   Person  Sales
    0  Google      Sam    200
    1  Google  Charlie    150
    



```python
dfGroup.get_group("Facebook")
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Company</th>
      <th>Person</th>
      <th>Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>Facebook</td>
      <td>Carl</td>
      <td>250</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Facebook</td>
      <td>Sarah</td>
      <td>350</td>
    </tr>
  </tbody>
</table>
</div>




```python
dfGroup.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">Sales</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>Company</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Apple</th>
      <td>2.0</td>
      <td>200.0</td>
      <td>141.421356</td>
      <td>100.0</td>
      <td>150.0</td>
      <td>200.0</td>
      <td>250.0</td>
      <td>300.0</td>
    </tr>
    <tr>
      <th>Facebook</th>
      <td>2.0</td>
      <td>300.0</td>
      <td>70.710678</td>
      <td>250.0</td>
      <td>275.0</td>
      <td>300.0</td>
      <td>325.0</td>
      <td>350.0</td>
    </tr>
    <tr>
      <th>Google</th>
      <td>2.0</td>
      <td>175.0</td>
      <td>35.355339</td>
      <td>150.0</td>
      <td>162.5</td>
      <td>175.0</td>
      <td>187.5</td>
      <td>200.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
dfGroup.describe().transpose()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Company</th>
      <th>Apple</th>
      <th>Facebook</th>
      <th>Google</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="8" valign="top">Sales</th>
      <th>count</th>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>200.000000</td>
      <td>300.000000</td>
      <td>175.000000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>141.421356</td>
      <td>70.710678</td>
      <td>35.355339</td>
    </tr>
    <tr>
      <th>min</th>
      <td>100.000000</td>
      <td>250.000000</td>
      <td>150.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>150.000000</td>
      <td>275.000000</td>
      <td>162.500000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>200.000000</td>
      <td>300.000000</td>
      <td>175.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>250.000000</td>
      <td>325.000000</td>
      <td>187.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>300.000000</td>
      <td>350.000000</td>
      <td>200.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
dfGroup.describe().transpose()["Apple"]
```




    Sales  count      2.000000
           mean     200.000000
           std      141.421356
           min      100.000000
           25%      150.000000
           50%      200.000000
           75%      250.000000
           max      300.000000
    Name: Apple, dtype: float64



## Merging, Joining and Concatenating


```python
df1 = pd.DataFrame({'A':['A0', 'A1', 'A2', 'A3'],
                    'B':['B0', 'B1', 'B2', 'B3'],
                    'C':['C0', 'C1', 'C2', 'C3'],
                    'D':['D0', 'D1', 'D2', 'D3'],
                   }, index = [0,1,2,3])

df2 = pd.DataFrame({'A':['A4', 'A5', 'A6', 'A7'],
                    'B':['B4', 'B5', 'B6', 'B7'],
                    'C':['C4', 'C5', 'C6', 'C7'],
                    'D':['D4', 'D5', 'D6', 'D7'],
                   }, index = [4,5,6,7])

df3 = pd.DataFrame({'A':['A8', 'A9', 'A10', 'A11'],
                    'B':['B8', 'B9', 'B10', 'B11'],
                    'C':['C8', 'C9', 'C10', 'C11'],
                    'D':['D8', 'D9', 'D10', 'D11'],
                   }, index = [8,9,10,11])
```


```python
pd.concat([df1, df2, df3])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A0</td>
      <td>B0</td>
      <td>C0</td>
      <td>D0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A1</td>
      <td>B1</td>
      <td>C1</td>
      <td>D1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A2</td>
      <td>B2</td>
      <td>C2</td>
      <td>D2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A3</td>
      <td>B3</td>
      <td>C3</td>
      <td>D3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A4</td>
      <td>B4</td>
      <td>C4</td>
      <td>D4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>A5</td>
      <td>B5</td>
      <td>C5</td>
      <td>D5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>A6</td>
      <td>B6</td>
      <td>C6</td>
      <td>D6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>A7</td>
      <td>B7</td>
      <td>C7</td>
      <td>D7</td>
    </tr>
    <tr>
      <th>8</th>
      <td>A8</td>
      <td>B8</td>
      <td>C8</td>
      <td>D8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>A9</td>
      <td>B9</td>
      <td>C9</td>
      <td>D9</td>
    </tr>
    <tr>
      <th>10</th>
      <td>A10</td>
      <td>B10</td>
      <td>C10</td>
      <td>D10</td>
    </tr>
    <tr>
      <th>11</th>
      <td>A11</td>
      <td>B11</td>
      <td>C11</td>
      <td>D11</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.concat([df1, df2, df3], axis=1)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A0</td>
      <td>B0</td>
      <td>C0</td>
      <td>D0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A1</td>
      <td>B1</td>
      <td>C1</td>
      <td>D1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A2</td>
      <td>B2</td>
      <td>C2</td>
      <td>D2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A3</td>
      <td>B3</td>
      <td>C3</td>
      <td>D3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>A4</td>
      <td>B4</td>
      <td>C4</td>
      <td>D4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>A5</td>
      <td>B5</td>
      <td>C5</td>
      <td>D5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>A6</td>
      <td>B6</td>
      <td>C6</td>
      <td>D6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>A7</td>
      <td>B7</td>
      <td>C7</td>
      <td>D7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>A8</td>
      <td>B8</td>
      <td>C8</td>
      <td>D8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>A9</td>
      <td>B9</td>
      <td>C9</td>
      <td>D9</td>
    </tr>
    <tr>
      <th>10</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>A10</td>
      <td>B10</td>
      <td>C10</td>
      <td>D10</td>
    </tr>
    <tr>
      <th>11</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>A11</td>
      <td>B11</td>
      <td>C11</td>
      <td>D11</td>
    </tr>
  </tbody>
</table>
</div>




```python
left = pd.DataFrame({'key':['K0','K1','K2','K3'],
                    'A':['A0','A1','A2','A3'],
                    'B':['B0','B1','B2','B3']})
left
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>K0</td>
      <td>A0</td>
      <td>B0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>K1</td>
      <td>A1</td>
      <td>B1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>K2</td>
      <td>A2</td>
      <td>B2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>K3</td>
      <td>A3</td>
      <td>B3</td>
    </tr>
  </tbody>
</table>
</div>




```python
right = pd.DataFrame({'key':['K0','K1','K2','K3'],
                     'C':['C0','C1','C2','C3'],
                     'D':['D0','D1','D2','D3']})
right
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>K0</td>
      <td>C0</td>
      <td>D0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>K1</td>
      <td>C1</td>
      <td>D1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>K2</td>
      <td>C2</td>
      <td>D2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>K3</td>
      <td>C3</td>
      <td>D3</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.merge(left, right, how='inner', on='key')
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>K0</td>
      <td>A0</td>
      <td>B0</td>
      <td>C0</td>
      <td>D0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>K1</td>
      <td>A1</td>
      <td>B1</td>
      <td>C1</td>
      <td>D1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>K2</td>
      <td>A2</td>
      <td>B2</td>
      <td>C2</td>
      <td>D2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>K3</td>
      <td>A3</td>
      <td>B3</td>
      <td>C3</td>
      <td>D3</td>
    </tr>
  </tbody>
</table>
</div>




```python
left = pd.DataFrame({'A':['A0','A1','A2','A3'],
                    'B':['B0','B1','B2','B3']
                    }, index = ['K0','K1','K2','K3'])
left
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>K0</th>
      <td>A0</td>
      <td>B0</td>
    </tr>
    <tr>
      <th>K1</th>
      <td>A1</td>
      <td>B1</td>
    </tr>
    <tr>
      <th>K2</th>
      <td>A2</td>
      <td>B2</td>
    </tr>
    <tr>
      <th>K3</th>
      <td>A3</td>
      <td>B3</td>
    </tr>
  </tbody>
</table>
</div>




```python
right = pd.DataFrame({'C':['C0','C2','C4','C5'],
                     'D':['D0','D2','D4','D5']
                     }, index = ['K0','K2','K4','K5'])
right
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>K0</th>
      <td>C0</td>
      <td>D0</td>
    </tr>
    <tr>
      <th>K2</th>
      <td>C2</td>
      <td>D2</td>
    </tr>
    <tr>
      <th>K4</th>
      <td>C4</td>
      <td>D4</td>
    </tr>
    <tr>
      <th>K5</th>
      <td>C5</td>
      <td>D5</td>
    </tr>
  </tbody>
</table>
</div>




```python
left.join(right)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>K0</th>
      <td>A0</td>
      <td>B0</td>
      <td>C0</td>
      <td>D0</td>
    </tr>
    <tr>
      <th>K1</th>
      <td>A1</td>
      <td>B1</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>K2</th>
      <td>A2</td>
      <td>B2</td>
      <td>C2</td>
      <td>D2</td>
    </tr>
    <tr>
      <th>K3</th>
      <td>A3</td>
      <td>B3</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
right.join(left)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>C</th>
      <th>D</th>
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>K0</th>
      <td>C0</td>
      <td>D0</td>
      <td>A0</td>
      <td>B0</td>
    </tr>
    <tr>
      <th>K2</th>
      <td>C2</td>
      <td>D2</td>
      <td>A2</td>
      <td>B2</td>
    </tr>
    <tr>
      <th>K4</th>
      <td>C4</td>
      <td>D4</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>K5</th>
      <td>C5</td>
      <td>D5</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
left.join(right, how='outer')
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>K0</th>
      <td>A0</td>
      <td>B0</td>
      <td>C0</td>
      <td>D0</td>
    </tr>
    <tr>
      <th>K1</th>
      <td>A1</td>
      <td>B1</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>K2</th>
      <td>A2</td>
      <td>B2</td>
      <td>C2</td>
      <td>D2</td>
    </tr>
    <tr>
      <th>K3</th>
      <td>A3</td>
      <td>B3</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>K4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>C4</td>
      <td>D4</td>
    </tr>
    <tr>
      <th>K5</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>C5</td>
      <td>D5</td>
    </tr>
  </tbody>
</table>
</div>



## Data Input and Output

### CSV File


```python
df = pd.read_csv('students.csv')
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>race/ethnicity</th>
      <th>parental level of education</th>
      <th>lunch</th>
      <th>test preparation course</th>
      <th>math score</th>
      <th>reading score</th>
      <th>writing score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>group B</td>
      <td>bachelor's degree</td>
      <td>standard</td>
      <td>none</td>
      <td>72</td>
      <td>72</td>
      <td>74</td>
    </tr>
    <tr>
      <th>1</th>
      <td>female</td>
      <td>group C</td>
      <td>some college</td>
      <td>standard</td>
      <td>completed</td>
      <td>69</td>
      <td>90</td>
      <td>88</td>
    </tr>
    <tr>
      <th>2</th>
      <td>female</td>
      <td>group B</td>
      <td>master's degree</td>
      <td>standard</td>
      <td>none</td>
      <td>90</td>
      <td>95</td>
      <td>93</td>
    </tr>
    <tr>
      <th>3</th>
      <td>male</td>
      <td>group A</td>
      <td>associate's degree</td>
      <td>free/reduced</td>
      <td>none</td>
      <td>47</td>
      <td>57</td>
      <td>44</td>
    </tr>
    <tr>
      <th>4</th>
      <td>male</td>
      <td>group C</td>
      <td>some college</td>
      <td>standard</td>
      <td>none</td>
      <td>76</td>
      <td>78</td>
      <td>75</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>female</td>
      <td>group E</td>
      <td>master's degree</td>
      <td>standard</td>
      <td>completed</td>
      <td>88</td>
      <td>99</td>
      <td>95</td>
    </tr>
    <tr>
      <th>996</th>
      <td>male</td>
      <td>group C</td>
      <td>high school</td>
      <td>free/reduced</td>
      <td>none</td>
      <td>62</td>
      <td>55</td>
      <td>55</td>
    </tr>
    <tr>
      <th>997</th>
      <td>female</td>
      <td>group C</td>
      <td>high school</td>
      <td>free/reduced</td>
      <td>completed</td>
      <td>59</td>
      <td>71</td>
      <td>65</td>
    </tr>
    <tr>
      <th>998</th>
      <td>female</td>
      <td>group D</td>
      <td>some college</td>
      <td>standard</td>
      <td>completed</td>
      <td>68</td>
      <td>78</td>
      <td>77</td>
    </tr>
    <tr>
      <th>999</th>
      <td>female</td>
      <td>group D</td>
      <td>some college</td>
      <td>free/reduced</td>
      <td>none</td>
      <td>77</td>
      <td>86</td>
      <td>86</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 8 columns</p>
</div>




```python
df.to_csv('example.csv', index=False)
```

### Excel file

#### pip install openpyxl


```python
xls = pd.ExcelFile('Excel_Sample.xlsx')
df = pd.read_excel(xls, 'Sheet1', index_col=0)

df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.read_excel(open('Excel_Sample.xlsx', 'rb'), sheet_name='Sheet1', index_col=0)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.to_excel('newExcel.xlsx', sheet_name='Sheet1')
```

#### HTML Files

#### lxml
#### html5lib
#### BeautifulSoup4


```python
df = pd.read_html('https://www.fdic.gov/resources/resolutions/bank-failures/failed-bank-list/')
df
```




    [                         Bank NameBank           CityCity StateSt  CertCert  \
     0                    Almena State Bank             Almena      KS     15426   
     1           First City Bank of Florida  Fort Walton Beach      FL     16748   
     2                 The First State Bank      Barboursville      WV     14361   
     3                   Ericson State Bank            Ericson      NE     18265   
     4     City National Bank of New Jersey             Newark      NJ     21111   
     ..                                 ...                ...     ...       ...   
     558                 Superior Bank, FSB           Hinsdale      IL     32646   
     559                Malta National Bank              Malta      OH      6629   
     560    First Alliance Bank & Trust Co.         Manchester      NH     34264   
     561  National State Bank of Metropolis         Metropolis      IL      3815   
     562                   Bank of Honolulu           Honolulu      HI     21029   
     
                      Acquiring InstitutionAI Closing DateClosing  FundFund  
     0                            Equity Bank    October 23, 2020     10538  
     1              United Fidelity Bank, fsb    October 16, 2020     10537  
     2                         MVB Bank, Inc.       April 3, 2020     10536  
     3             Farmers and Merchants Bank   February 14, 2020     10535  
     4                        Industrial Bank    November 1, 2019     10534  
     ..                                   ...                 ...       ...  
     558                Superior Federal, FSB       July 27, 2001      6004  
     559                    North Valley Bank         May 3, 2001      4648  
     560  Southern New Hampshire Bank & Trust    February 2, 2001      4647  
     561              Banterra Bank of Marion   December 14, 2000      4646  
     562                   Bank of the Orient    October 13, 2000      4645  
     
     [563 rows x 7 columns]]




```python
df[0]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bank NameBank</th>
      <th>CityCity</th>
      <th>StateSt</th>
      <th>CertCert</th>
      <th>Acquiring InstitutionAI</th>
      <th>Closing DateClosing</th>
      <th>FundFund</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Almena State Bank</td>
      <td>Almena</td>
      <td>KS</td>
      <td>15426</td>
      <td>Equity Bank</td>
      <td>October 23, 2020</td>
      <td>10538</td>
    </tr>
    <tr>
      <th>1</th>
      <td>First City Bank of Florida</td>
      <td>Fort Walton Beach</td>
      <td>FL</td>
      <td>16748</td>
      <td>United Fidelity Bank, fsb</td>
      <td>October 16, 2020</td>
      <td>10537</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The First State Bank</td>
      <td>Barboursville</td>
      <td>WV</td>
      <td>14361</td>
      <td>MVB Bank, Inc.</td>
      <td>April 3, 2020</td>
      <td>10536</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ericson State Bank</td>
      <td>Ericson</td>
      <td>NE</td>
      <td>18265</td>
      <td>Farmers and Merchants Bank</td>
      <td>February 14, 2020</td>
      <td>10535</td>
    </tr>
    <tr>
      <th>4</th>
      <td>City National Bank of New Jersey</td>
      <td>Newark</td>
      <td>NJ</td>
      <td>21111</td>
      <td>Industrial Bank</td>
      <td>November 1, 2019</td>
      <td>10534</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>558</th>
      <td>Superior Bank, FSB</td>
      <td>Hinsdale</td>
      <td>IL</td>
      <td>32646</td>
      <td>Superior Federal, FSB</td>
      <td>July 27, 2001</td>
      <td>6004</td>
    </tr>
    <tr>
      <th>559</th>
      <td>Malta National Bank</td>
      <td>Malta</td>
      <td>OH</td>
      <td>6629</td>
      <td>North Valley Bank</td>
      <td>May 3, 2001</td>
      <td>4648</td>
    </tr>
    <tr>
      <th>560</th>
      <td>First Alliance Bank &amp; Trust Co.</td>
      <td>Manchester</td>
      <td>NH</td>
      <td>34264</td>
      <td>Southern New Hampshire Bank &amp; Trust</td>
      <td>February 2, 2001</td>
      <td>4647</td>
    </tr>
    <tr>
      <th>561</th>
      <td>National State Bank of Metropolis</td>
      <td>Metropolis</td>
      <td>IL</td>
      <td>3815</td>
      <td>Banterra Bank of Marion</td>
      <td>December 14, 2000</td>
      <td>4646</td>
    </tr>
    <tr>
      <th>562</th>
      <td>Bank of Honolulu</td>
      <td>Honolulu</td>
      <td>HI</td>
      <td>21029</td>
      <td>Bank of the Orient</td>
      <td>October 13, 2000</td>
      <td>4645</td>
    </tr>
  </tbody>
</table>
<p>563 rows × 7 columns</p>
</div>




```python
df = pd.DataFrame({'name':['Sam','Charlie','Amy','Sally','Carl','Sarah'],
               'physics':[70, 60, 65, 90, 80, 95],
               'chemistry':[85, 80, 90, 85, 80, 100],
               'algebra':[80, 90, 85, 90, 60, 85]})

```


```python
html = df.to_html()

textFile = open('index.html', 'w')
textFile.write(html)
textFile.close()
```
