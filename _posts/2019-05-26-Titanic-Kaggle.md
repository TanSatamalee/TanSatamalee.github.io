---
layout: post
title: Titanic Kaggle Analysis
---

Hello to who ever is reading this! This will be my first blog ever so be warned! For my first, I chose to work with the [Titanic Kaggle Dataset](https://www.kaggle.com/c/titanic) and do a straigthforward analysis with random forests after cleaning and processing it.

We first load in the two datasets we are going to be working with.

```python
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
```

After importing the data, we first take a glance and look at what we are working with.

```python
train.describe()
```

IMAGE

```python
train.describe()
```
IMAGE

```python
train.info()
```

IMAGE

The data consists of 891 points with 7 different features. Most notably, we will see that there are missing values for Age (177 values), Cabin (687 values), and Embarked (2 values). Also, looking at the Fare summary, we see that the minimum is 0. That mean either some people are getting in for free or those are also missing values.

Let's first convert everything to numerical values, so that it's much easier for us to work with. From the info description, we see that Name, Sex, Ticket, Cabin, and Embarked are all non-numeric values, so we will convert them using the following block of code:

```python
def convert_embarked(x):
    if x == 'S':
        return 2
    if x == 'C':
        return 1
    if x == 'Q':
        return 3

def convert_sex(x):
    if x == 'male':
        return 1
    if x == 'female':
        return 0

def convert_ticket(x):
    x = x.split()[-1]
    if x.isnumeric():
        return int(x)
    return -1

def convert_cabin(x):
    x = x[0]
    if x == 'A':
        return 1
    if x == 'B':
        return 2
    if x == 'C':
        return 3
    if x == 'D':
        return 4
    if x == 'E':
        return 5
    return -1
    
def convert_title(x):
    if x == 'Capt.':
        return 1
    if x in ['Col.', 'Major.']:
        return 2
    if x in ['Rev.', 'Dr.']:
        return 3
    if x in ['Mr.', 'Sir.', 'Don.', 'Jonkheer.']:
        return 4
    if x == 'Master.':
        return 5
    if x in ['Miss.', 'Mlle.', 'Mme.', 'Ms.', 'Lady.']:
        return 6
    if x == 'Mrs.':
        return 7
        
train.Embarked = train.Embarked.apply(convert_embarked)
train.Sex = train.Sex.apply(convert_sex)
train.Ticket = train.Ticket.apply(convert_ticket)
train.Cabin.fillna('None', inplace=True)
train.Cabin = train.Cabin.apply(convert_cabin)
train['Title'] = train.Name.str.extract('\, ([A-Z][^ ]*\.)',expand=False)
train.Title = train.Title.apply(convert_title)
```

