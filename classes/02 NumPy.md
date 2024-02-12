# NumPy

## Using NumPy


```python
import numpy as np
```

https://numpy.org/doc/stable/index.html

## NumPy Arrays


```python
myList = [1, 2, 3, 4, 5]
npList = np.array(myList)
npList
```




    array([1, 2, 3, 4, 5])




```python
myMatrix = [[1, 2, 3],[4, 5, 6], [7, 8, 9]]
npMatrix = np.array(myMatrix)
npMatrix
```




    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])



## Built in Functions

### arange


```python
#range(n) -> 0 1 2 3 4 ... n-1
np.arange(0, 10)
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
# start stop step
np.arange(0, 100, 10)
```




    array([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90])



### zeros


```python
#np.zeros(length vector)
np.zeros(5)
```




    array([0., 0., 0., 0., 0.])




```python
np.zeros((3,4))
```




    array([[0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]])



### ones


```python
np.ones(5)
```




    array([1., 1., 1., 1., 1.])




```python
np.ones((3,4))
```




    array([[1., 1., 1., 1.],
           [1., 1., 1., 1.],
           [1., 1., 1., 1.]])



### identity


```python
np.identity(5)
```




    array([[1., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0.],
           [0., 0., 1., 0., 0.],
           [0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 1.]])



### eye


```python
np.eye(5, k=2)
```




    array([[0., 0., 1., 0., 0.],
           [0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 1.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.]])




```python
np.eye(10, k=-4)
```




    array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]])



### linspace


```python
#np.linspace(start, stop, count) -> stop is included.
np.linspace(0, 10, 7)
```




    array([ 0.        ,  1.66666667,  3.33333333,  5.        ,  6.66666667,
            8.33333333, 10.        ])




```python
np.linspace(0, 10, 7, endpoint=False)
```




    array([0.        , 1.42857143, 2.85714286, 4.28571429, 5.71428571,
           7.14285714, 8.57142857])



### random


```python
#Between 0 and 1
np.random.rand(5)
```




    array([0.90930226, 0.89425991, 0.11427288, 0.1273502 , 0.88187837])




```python
np.random.rand(3,4)
```




    array([[0.54321034, 0.33165273, 0.34603816, 0.98359018],
           [0.37470272, 0.30314285, 0.67115931, 0.68833785],
           [0.20105748, 0.8535962 , 0.00454698, 0.74155848]])



### randn


```python
#Rand random n normal (Gaussian)
#mean 0 variance 1
np.random.randn(5)
```




    array([ 2.75917803, -1.02535276,  1.01654945, -0.05281583,  0.31444278])




```python
np.random.randn(3,4)
```




    array([[ 0.64458968, -0.4673063 ,  2.04540563,  0.23234617],
           [ 1.2991192 ,  0.26371628, -0.85254662, -1.11468169],
           [ 1.66246049, -0.52597445,  1.44761443,  0.10514107]])



### randint


```python
np.random.randint(1, 10)
```




    4




```python
"""
randint(lower, upper, count) 
-> lower is included
-> upper is excluded
"""
np.random.randint(1, 100, 10)
```




    array([73, 68, 53, 95, 26, 30, 54, 40, 63,  4])



### reshape


```python
#seed
#random_state
np.random.seed(50)
```


```python
myArray = np.random.rand(25)
myArray
```




    array([0.49460165, 0.2280831 , 0.25547392, 0.39632991, 0.3773151 ,
           0.99657423, 0.4081972 , 0.77189399, 0.76053669, 0.31000935,
           0.3465412 , 0.35176482, 0.14546686, 0.97266468, 0.90917844,
           0.5599571 , 0.31359075, 0.88820004, 0.67457307, 0.39108745,
           0.50718412, 0.5241035 , 0.92800093, 0.57137307, 0.66833757])




```python
reArray = myArray.reshape(5,5)
reArray
```




    array([[0.49460165, 0.2280831 , 0.25547392, 0.39632991, 0.3773151 ],
           [0.99657423, 0.4081972 , 0.77189399, 0.76053669, 0.31000935],
           [0.3465412 , 0.35176482, 0.14546686, 0.97266468, 0.90917844],
           [0.5599571 , 0.31359075, 0.88820004, 0.67457307, 0.39108745],
           [0.50718412, 0.5241035 , 0.92800093, 0.57137307, 0.66833757]])



### min, max, argmin, argmax


```python
reArray.min()
```




    0.14546685649615498




```python
reArray.max()
```




    0.9965742301546493




```python
# 1: 2
# 50: 12
# 100: 4
reArray.argmin()
```




    12




```python
np.unravel_index(reArray.argmin(), (5,5))
```




    (2, 2)




```python
reArray.argmax()
```




    5



### shape


```python
myArray.shape
```




    (25,)




```python
myArray = myArray.reshape(1,25)
print(myArray.shape)
myArray
```

    (1, 25)





    array([[0.49460165, 0.2280831 , 0.25547392, 0.39632991, 0.3773151 ,
            0.99657423, 0.4081972 , 0.77189399, 0.76053669, 0.31000935,
            0.3465412 , 0.35176482, 0.14546686, 0.97266468, 0.90917844,
            0.5599571 , 0.31359075, 0.88820004, 0.67457307, 0.39108745,
            0.50718412, 0.5241035 , 0.92800093, 0.57137307, 0.66833757]])




```python
myArray = myArray.reshape(25,1)
print(myArray.shape)
myArray
```

    (25, 1)





    array([[0.49460165],
           [0.2280831 ],
           [0.25547392],
           [0.39632991],
           [0.3773151 ],
           [0.99657423],
           [0.4081972 ],
           [0.77189399],
           [0.76053669],
           [0.31000935],
           [0.3465412 ],
           [0.35176482],
           [0.14546686],
           [0.97266468],
           [0.90917844],
           [0.5599571 ],
           [0.31359075],
           [0.88820004],
           [0.67457307],
           [0.39108745],
           [0.50718412],
           [0.5241035 ],
           [0.92800093],
           [0.57137307],
           [0.66833757]])




```python
myArray = myArray.reshape(5,5)
print(myArray.shape)
myArray
```

    (5, 5)





    array([[0.49460165, 0.2280831 , 0.25547392, 0.39632991, 0.3773151 ],
           [0.99657423, 0.4081972 , 0.77189399, 0.76053669, 0.31000935],
           [0.3465412 , 0.35176482, 0.14546686, 0.97266468, 0.90917844],
           [0.5599571 , 0.31359075, 0.88820004, 0.67457307, 0.39108745],
           [0.50718412, 0.5241035 , 0.92800093, 0.57137307, 0.66833757]])



### Indexing


```python
myArray = np.array([1, 2, 3, 4, 5])
myArray
```




    array([1, 2, 3, 4, 5])




```python
myArray[2]
```

    3




```python
myArray[-1]
```

    5




```python
myMatrix = np.array([[1,2,3],[4,5,6],[7,8,9]])
myMatrix
```


    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])




```python
myMatrix[1,2]
```


    6



### row and column


```python
print(npMatrix[1])
```

    [4 5 6]



```python
print(npMatrix[:,2])
```

    [3 6 9]



```python
print(npMatrix[:,2].reshape(3,1))
```

    [[3]
     [6]
     [9]]


### Indexing in 3D


```python
m = np.array([[[10,11,12],[13,14,15],[16,17,18]],
              [[20,21,22],[23,24,25],[26,27,28]],
              [[30,31,32],[33,34,35],[36,37,38]]])
```


```python
#5D 4D matrix row column
print(m[2, 0, 1])
```

    31



```python
print(m[1, 2])
```

    [26 27 28]



```python
print(m[0, :, 1])
```

    [11 14 17]



```python
print(m[:,1,2])
```

    [15 25 35]



```python
print(m[1])
```

    [[20 21 22]
     [23 24 25]
     [26 27 28]]



```python
print(m[:, 2])
```

    [[16 17 18]
     [26 27 28]
     [36 37 38]]



```python
print(m[:,:,0])
```

    [[10 13 16]
     [20 23 26]
     [30 33 36]]


### Slicing


```python
x = np.arange(0,6)
x
```




    array([0, 1, 2, 3, 4, 5])




```python
#slice[start, stop] start is included, but stop is not.
y = x[1:4]
y
```




    array([1, 2, 3])



### Broadcasting


```python
x[:] = 10
x
```




    array([10, 10, 10, 10, 10, 10])



### Operations


```python
myArray = np.arange(0,10)
myArray
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
myArray + myArray
```




    array([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18])




```python
myArray ** 2
```




    array([ 0,  1,  4,  9, 16, 25, 36, 49, 64, 81])




```python
myArray // 2
```




    array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])




```python
myArray / 3
```




    array([0.        , 0.33333333, 0.66666667, 1.        , 1.33333333,
           1.66666667, 2.        , 2.33333333, 2.66666667, 3.        ])




```python
#NaN = Not a Number
nanArray = myArray / myArray
nanArray
```




    array([nan,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])




```python
nanArray == 'nan'
```



    False




```python
nanArray == np.nan
```




    array([False, False, False, False, False, False, False, False, False,
           False])




```python
type(np.nan) == type(nanArray[0])
```




    False




```python
np.isnan(nanArray)
```




    array([ True, False, False, False, False, False, False, False, False,
           False])




```python
#inf = infinity
1 / myArray
```





    array([       inf, 1.        , 0.5       , 0.33333333, 0.25      ,
           0.2       , 0.16666667, 0.14285714, 0.125     , 0.11111111])



### Universal Array functions


```python
np.sqrt(myArray)
```




    array([0.        , 1.        , 1.41421356, 1.73205081, 2.        ,
           2.23606798, 2.44948974, 2.64575131, 2.82842712, 3.        ])




```python
np.cos(myArray)
```




    array([ 1.        ,  0.54030231, -0.41614684, -0.9899925 , -0.65364362,
            0.28366219,  0.96017029,  0.75390225, -0.14550003, -0.91113026])




```python
np.log(myArray)
```




    array([      -inf, 0.        , 0.69314718, 1.09861229, 1.38629436,
           1.60943791, 1.79175947, 1.94591015, 2.07944154, 2.19722458])




```python
np.negative(myArray) #sum symetric
```




    array([ 0, -1, -2, -3, -4, -5, -6, -7, -8, -9])




```python
np.exp2(myArray)
```




    array([  1.,   2.,   4.,   8.,  16.,  32.,  64., 128., 256., 512.])


