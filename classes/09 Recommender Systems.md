# 09 Recommender Systems


```python
import pandas as pd

ratings = pd.read_csv("https://s3-us-west-2.amazonaws.com/recommender-tutorial/ratings.csv")
ratings.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>4.0</td>
      <td>964982703</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>3</td>
      <td>4.0</td>
      <td>964981247</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>6</td>
      <td>4.0</td>
      <td>964982224</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>47</td>
      <td>5.0</td>
      <td>964983815</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>50</td>
      <td>5.0</td>
      <td>964982931</td>
    </tr>
  </tbody>
</table>
</div>




```python
movies = pd.read_csv("https://s3-us-west-2.amazonaws.com/recommender-tutorial/movies.csv")
movies.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama|Romance</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
    </tr>
  </tbody>
</table>
</div>




```python
n_movies = len(ratings['movieId'].unique())
```


```python
n_ratings = len(ratings)
n_users = len(ratings['userId'].unique())

print(f"Number of ratings: {n_ratings}")
print(f"Number of unique movieId's: {n_movies}")
print(f"Number of unique users: {n_users}")
print(f"Average ratings per user: {round(n_ratings/n_users, 2)}")
print(f"Average ratings per movie: {round(n_ratings/n_movies, 2)}")
```

    Number of ratings: 100836
    Number of unique movieId's: 9724
    Number of unique users: 610
    Average ratings per user: 165.3
    Average ratings per movie: 10.37



```python
user_freq = ratings[['userId', 'movieId']].groupby('userId').count().reset_index()
user_freq.columns = ['userId', 'movieId']
user_freq.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>232</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>29</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>39</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>216</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>44</td>
    </tr>
  </tbody>
</table>
</div>



## Simple Recommender


```python
mean_rating = ratings.groupby('movieId')[['rating']].mean()

lowest_rated = mean_rating['rating'].idxmin()
movies.loc[movies['movieId'] == lowest_rated]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2689</th>
      <td>3604</td>
      <td>Gypsy (1962)</td>
      <td>Musical</td>
    </tr>
  </tbody>
</table>
</div>




```python
highest_rated = mean_rating['rating'].idxmax()
movies.loc[movies['movieId'] == highest_rated]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>48</th>
      <td>53</td>
      <td>Lamerica (1994)</td>
      <td>Adventure|Drama</td>
    </tr>
  </tbody>
</table>
</div>




```python
ratings[ratings['movieId']==lowest_rated]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13633</th>
      <td>89</td>
      <td>3604</td>
      <td>0.5</td>
      <td>1520408880</td>
    </tr>
  </tbody>
</table>
</div>




```python
ratings[ratings['movieId']==highest_rated]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13368</th>
      <td>85</td>
      <td>53</td>
      <td>5.0</td>
      <td>889468268</td>
    </tr>
    <tr>
      <th>96115</th>
      <td>603</td>
      <td>53</td>
      <td>5.0</td>
      <td>963180003</td>
    </tr>
  </tbody>
</table>
</div>



## Collaborative filter


```python
N = len(ratings['userId'].unique())
M = len(ratings['movieId'].unique())
```


```python
import numpy as np

user_mapper = dict(zip(np.unique(ratings["userId"]), list(range(N))))
movie_mapper = dict(zip(np.unique(ratings["movieId"]), list(range(M))))
```


```python
user_index = [user_mapper[i] for i in ratings['userId']]
movie_index = [movie_mapper[i] for i in ratings['movieId']]
```


```python
from scipy.sparse import csr_matrix #Compressed Sparse Rows

X = csr_matrix((ratings["rating"], (movie_index, user_index)), shape=(M, N))

X_df =pd.DataFrame(X.toarray())
X_df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>600</th>
      <th>601</th>
      <th>602</th>
      <th>603</th>
      <th>604</th>
      <th>605</th>
      <th>606</th>
      <th>607</th>
      <th>608</th>
      <th>609</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>4.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>2.5</td>
      <td>4.0</td>
      <td>2.5</td>
      <td>3.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>3.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <th>9719</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9720</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9721</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9722</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9723</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>9724 rows × 610 columns</p>
</div>




```python
import random

movie_id = random.randint(0, n_movies)
movies.loc[movies['movieId']==ratings['movieId'].unique()[movie_id]]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2216</th>
      <td>2944</td>
      <td>Dirty Dozen, The (1967)</td>
      <td>Action|Drama|War</td>
    </tr>
  </tbody>
</table>
</div>




```python
import sklearn

from sklearn.neighbors import NearestNeighbors
"""
Find similar movies using KNN
"""
neighbor_ids = []
      
movie_ind = movie_mapper[movie_id]
movie_vec = X[movie_ind]
k=10
kNN = NearestNeighbors(n_neighbors=k+1, algorithm="brute", metric="cosine")
kNN.fit(X)
movie_vec = movie_vec.reshape(1,-1)
neighbor = kNN.kneighbors(movie_vec, return_distance=False)

movie_inv_mapper = dict(zip(list(range(M)), np.unique(ratings["movieId"])))
    
for i in range(0,k):
    n = neighbor.item(i)
    neighbor_ids.append(movie_inv_mapper[n])
neighbor_ids.pop(0)
neighbor_ids
```




    [4275, 3053, 2097, 4849, 7354, 4636, 2766, 448, 1609]




```python
def find_similar_movies(movie_id, X, k):
      
    neighbour_ids = []
      
    movie_ind = movie_mapper[movie_id]
    movie_vec = X[movie_ind]
    k+=1
    kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric='cosine')
    kNN.fit(X)
    movie_vec = movie_vec.reshape(1,-1)
    neighbor = kNN.kneighbors(movie_vec, return_distance=False)
    for i in range(0,k):
        n = neighbor.item(i)
        neighbor_ids.append(movie_inv_mapper[n])
    neighbor_ids.pop(0)
    return neighbor_ids
```


```python
movie_titles = dict(zip(movies['movieId'], movies['title']))
#movie_titles
```


```python
#586
movie_id = 586
movie_title = movie_titles[movie_id]
movie_title
```




    'Home Alone (1990)'




```python
similar_ids = find_similar_movies(movie_id, X, k=10)

print(f"Since you watched {movie_title}")
for i in similar_ids:
    print(movie_titles[i])
```

    Since you watched Home Alone (1990)
    Messenger: The Story of Joan of Arc, The (1999)
    Something Wicked This Way Comes (1983)
    My First Mister (2001)
    Mad Dog and Glory (1993)
    Punisher, The (1989)
    Adventures of Sebastian Cole, The (1998)
    Fearless (1993)
    187 (One Eight Seven) (1997)
    Home Alone (1990)
    Mrs. Doubtfire (1993)
    Lion King, The (1994)
    Pretty Woman (1990)
    Jurassic Park (1993)
    Jumanji (1995)
    Speed (1994)
    Forrest Gump (1994)
    Aladdin (1992)
    Mask, The (1994)
    Indiana Jones and the Temple of Doom (1984)



```python
rated = ratings.loc[ratings["movieId"]==movie_id]
rated = rated.loc[rated["rating"]>3]#

print(rated)
```

           userId  movieId  rating   timestamp
    1820       18      586     3.5  1455748696
    8803       62      586     4.0  1521489913
    15558     102      586     4.0   835877270
    15604     103      586     4.0  1431957135
    18280     116      586     3.5  1337199910
    18462     117      586     4.0   844162913
    24357     169      586     5.0  1078284644
    26192     182      586     3.5  1063289220
    32077     220      586     4.5  1230061714
    33116     226      586     4.0  1095662748
    33754     230      586     3.5  1196304782
    39337     274      586     3.5  1171759271
    41264     280      586     4.0  1348532002
    43523     292      586     4.0  1265680476
    45794     304      586     4.0   891173994
    49989     322      586     3.5  1217676382
    51007     330      586     3.5  1285905186
    52591     344      586     5.0  1420496646
    54172     357      586     4.0  1348612117
    56955     380      586     4.0  1494803432
    61301     402      586     4.0   849601295
    62126     411      586     4.0   835532644
    67450     436      586     4.0   833530512
    75207     475      586     4.5  1498031682
    76317     480      586     4.0  1179177983
    77978     484      586     4.0  1342296049
    79236     491      586     5.0  1526673066
    81104     514      586     3.5  1533873619
    81568     517      586     5.0  1487954357
    82002     520      586     4.0  1326609231
    82646     525      586     3.5  1476480345
    83606     534      586     4.5  1459787997
    85370     555      586     4.0   978747175
    86624     561      586     3.5  1491095448
    88705     573      586     5.0  1186590114
    89977     584      586     4.0   834988150
    91378     592      586     4.0   837350242
    91542     594      586     5.0  1109036952



```python
df = X_df.T.copy()
df['userBias'] = df[df!=0].mean(numeric_only=True, axis=1)
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>9715</th>
      <th>9716</th>
      <th>9717</th>
      <th>9718</th>
      <th>9719</th>
      <th>9720</th>
      <th>9721</th>
      <th>9722</th>
      <th>9723</th>
      <th>userBias</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.366379</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.948276</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.435897</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.555556</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.636364</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <th>605</th>
      <td>2.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.657399</td>
    </tr>
    <tr>
      <th>606</th>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.786096</td>
    </tr>
    <tr>
      <th>607</th>
      <td>2.5</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.134176</td>
    </tr>
    <tr>
      <th>608</th>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.270270</td>
    </tr>
    <tr>
      <th>609</th>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.688556</td>
    </tr>
  </tbody>
</table>
<p>610 rows × 9725 columns</p>
</div>




```python
rated["userId"].values
```




    array([ 18,  62, 102, 103, 116, 117, 169, 182, 220, 226, 230, 274, 280,
           292, 304, 322, 330, 344, 357, 380, 402, 411, 436, 475, 480, 484,
           491, 514, 517, 520, 525, 534, 555, 561, 573, 584, 592, 594])




```python
rated = ratings.loc[ratings["movieId"]==movie_id]
usrBias = df.iloc[rated["userId"].values]["userBias"]
usrBias = usrBias.reset_index(drop=True)
usrBias
```




    0      3.260870
    1      3.448148
    2      2.607397
    3      3.590909
    4      3.260722
             ...   
    111    3.266990
    112    4.200000
    113    2.991481
    114    3.507953
    115    3.270270
    Name: userBias, Length: 116, dtype: float64




```python

rated = rated.reset_index(drop=True)
rated
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8</td>
      <td>586</td>
      <td>3.0</td>
      <td>839463702</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14</td>
      <td>586</td>
      <td>3.0</td>
      <td>835441451</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18</td>
      <td>586</td>
      <td>3.5</td>
      <td>1455748696</td>
    </tr>
    <tr>
      <th>3</th>
      <td>19</td>
      <td>586</td>
      <td>3.0</td>
      <td>965707079</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20</td>
      <td>586</td>
      <td>3.0</td>
      <td>1054038279</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>111</th>
      <td>592</td>
      <td>586</td>
      <td>4.0</td>
      <td>837350242</td>
    </tr>
    <tr>
      <th>112</th>
      <td>594</td>
      <td>586</td>
      <td>5.0</td>
      <td>1109036952</td>
    </tr>
    <tr>
      <th>113</th>
      <td>599</td>
      <td>586</td>
      <td>3.0</td>
      <td>1498525239</td>
    </tr>
    <tr>
      <th>114</th>
      <td>602</td>
      <td>586</td>
      <td>1.0</td>
      <td>840875757</td>
    </tr>
    <tr>
      <th>115</th>
      <td>608</td>
      <td>586</td>
      <td>1.0</td>
      <td>1117504351</td>
    </tr>
  </tbody>
</table>
<p>116 rows × 4 columns</p>
</div>




```python
filtering = rated["rating"]>=usrBias
filtering
```




    0      False
    1      False
    2       True
    3      False
    4      False
           ...  
    111     True
    112     True
    113     True
    114    False
    115    False
    Length: 116, dtype: bool




```python
recomend = rated.loc[filtering]

print(recomend)
```

         userId  movieId  rating   timestamp
    2        18      586     3.5  1455748696
    10       62      586     4.0  1521489913
    19      102      586     4.0   835877270
    20      103      586     4.0  1431957135
    23      116      586     3.5  1337199910
    24      117      586     4.0   844162913
    30      169      586     5.0  1078284644
    38      220      586     4.5  1230061714
    40      229      586     3.0   838143590
    50      280      586     4.0  1348532002
    54      292      586     4.0  1265680476
    56      304      586     4.0   891173994
    59      322      586     3.5  1217676382
    62      344      586     5.0  1420496646
    65      357      586     4.0  1348612117
    68      380      586     4.0  1494803432
    73      402      586     4.0   849601295
    74      411      586     4.0   835532644
    79      436      586     4.0   833530512
    85      475      586     4.5  1498031682
    87      477      586     3.0  1200939829
    88      480      586     4.0  1179177983
    90      484      586     4.0  1342296049
    92      491      586     5.0  1526673066
    96      517      586     5.0  1487954357
    97      520      586     4.0  1326609231
    102     534      586     4.5  1459787997
    107     573      586     5.0  1186590114
    111     592      586     4.0   837350242
    112     594      586     5.0  1109036952
    113     599      586     3.0  1498525239
