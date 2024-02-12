# Introduction to Python



```python
#My comments
print("Hello World.")
```

    Hello World.



```python
print("Hello World.") #My comments
```

    Hello World.



```python
"""
My comments
for multiple
lines
"""


# I 
# can 
# use
# comments

# cmd + /
# crtl + /
```




    '\nMy comments\nfor multiple\nlines\n'



## Variables


```python
x = 5
y = 'string'

print(x, y)
```

    5 string



```python
x = 10
x = 'another'

print(x, y)
```

    another string


### Rules for Variables


```python
myVar = 1
my_var = 1
_myVar = 1
myvar2 = 1
MyVar = 10

print(myVar)
MyVar
```

    1

    10


```python
2myvar = 1
```


      File "/var/folders/tw/v62mplpd0cn_vt03pg_cm3940000gn/T/ipykernel_2189/4079902239.py", line 1
        2myvar = 1
             ^
    SyntaxError: invalid syntax




```python
x, y, z = 100, "type", True

print(x, y, z)
```

    100 type True



```python
x = 5

def func():
    global x
    print("Value of x is ", x)
    x = 1
    
    
func()
func()
x
```

    Value of x is  5
    Value of x is  1

    1



## Printing


```python
myVar = "something"

myVar
```




    'something'




```python
print(myVar)
```

    something



```python
name = "Jose"
age = 34

print("My name is {var1} and I am {var2} years old.".format(var1=name, var2=age))
print("My name is {} and I am {} years old.".format(name, age))
```

    My name is Jose and I am 34 years old.
    My name is Jose and I am 34 years old.


## Data Types


```python
z = 2/3
type(z)
```




    float




```python
x = False
type(x)
```




    bool




```python
type(z)==type(x)
```




    False




```python
x = int(z)

print(x, type(x))
```

    0 <class 'int'>


## Strings


```python
'single'
```




    'single'




```python
"double"
```




    'double'




```python
phrase = "My name is "

print(phrase+name)
print("My age is", age)
```

    My name is Jose
    My age is 34



```python
words = "You're students."
words
```




    "You're students."



## Lists


```python
['a', 'b', 'c', 'd', 'e']
```




    ['a', 'b', 'c', 'd', 'e']




```python
myList = [10, 6.5, "c", ["You", "Me"], True]
myList
```




    [10, 6.5, 'c', ['You', 'Me'], True]




```python
myList[-1]
```




    True




```python
myList[1:4]
```




    [6.5, 'c', ['You', 'Me']]




```python
b = False
myList.append(b)
myList
```




    [10, 6.5, 'c', ['You', 'Me'], True, False]




```python
myList = myList[0:4]
myList
```




    [10, 6.5, 'c', ['You', 'Me']]




```python
myList[2:]
```




    ['c', ['You', 'Me']]




```python
len(myList[3])
```




    2




```python
myList.insert(1, 'y')
myList
```




    [10, 'y', 6.5, 'c', ['You', 'Me']]




```python
myList.remove(["You", "Me"])
myList
```




    [10, 'y', 6.5, 'c']




```python
myList.pop(-1)
myList
```




    [10, 'y', 6.5]




```python
del myList[0]
myList
```




    ['y', 6.5]




```python
mySecList = [1,2,3]
list3 = myList + mySecList
list3
```




    ['y', 6.5, 1, 2, 3]




```python
print(*list3)
```

    y 6.5 1 2 3


## Dictionaries


```python
ids = {
    1:{ #index
        "name": "Jose", #key : item
        "age": "34",
        "height": "187",
        "year": "2019"
    },
    2:{ #index
        "name": "Ana", #key : item
        "age": "36",
        "height": "167",
        "year": "2017"
    }   
}

ids
```




    {1: {'name': 'Jose', 'age': '34', 'height': '187', 'year': '2019'},
     2: {'name': 'Ana', 'age': '36', 'height': '167', 'year': '2017'}}




```python
myName = ids[1]
other = ids.get(2)

print(myName)
print(other)
```

    {'name': 'Jose', 'age': '34', 'height': '187', 'year': '2019'}
    {'name': 'Ana', 'age': '36', 'height': '167', 'year': '2017'}



```python
age = other["age"]
age
```




    '36'




```python
myName["year"] = 1987
myName
```




    {'name': 'Jose', 'age': '34', 'height': '187', 'year': 1987}



## Sets


```python
set1 = {2, "a", 3}
set2 = {"b", 1, "c", 3}

set3 = set1.union(set2)
set3
```




    {1, 2, 3, 'a', 'b', 'c'}




```python
set3.discard(3)
set3
```




    {1, 2, 'a', 'b', 'c'}




```python
set3.discard(3)
set3
```




    {1, 2, 'a', 'b', 'c'}




```python
set3.remove("a")
set3
```




    {1, 2, 'b', 'c'}




```python
set3.remove("a")
set3
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    /var/folders/tw/v62mplpd0cn_vt03pg_cm3940000gn/T/ipykernel_2189/3391087153.py in <module>
    ----> 1 set3.remove("a")
          2 set3


    KeyError: 'a'


## Operators


```python
x = 10
y = 20
```


```python
x > y
```




    False




```python
x < y
```




    True




```python
x == y
```




    False




```python
x >= y
```




    False




```python
x <= y
```




    True




```python
#not(x == y)
(x != y)
```




    True




```python
x > 5 and y != 10 #or
```




    True



## if statements


```python
if x < y:
    print("is less")
```

    is less



```python
x = 25
if  x < y:
    print("is less")
else: 
    print("is greater")
```

    is greater



```python
x = 20
if x < y:
    print("is less")
elif x > y:
    print("is greater")
else:
    print("is equal")
```

    is equal


## Loops


```python
newSet = range(10)
newSet
```




    range(0, 10)




```python
for i in range(10):
    print(i)
```

    0
    1
    2
    3
    4
    5
    6
    7
    8
    9



```python
for i in range(4, 10, 2):
    print(i)
```

    4
    6
    8



```python
import random
i=0
while i<28:
    i = random.randint(1, 30)
    print(i)
```

    17
    1
    25
    7
    20
    22
    24
    28
