---
layout: post
title: Titanic Kaggle Analysis
---

Hello to who ever is reading this! This will be my first blog ever so be warned! For my first, I chose to work with the [Titanic Kaggle Dataset](https://www.kaggle.com/c/titanic) and do a straigthforward analysis with random forests after cleaning and processing it.

We first load in the two datasets we are going to be working with.

`python
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
`

After importing the data, we first take a glance and look at what we are working with.

`python
train.describe()
`

IMAGE

`python
train.describe()
`
IMAGE

`python
train.info()
`

IMAGE

The data consists of 891 points with 7 different features. Most notably, we will see that there are missing values for Age (177 values), Cabin (687 values), and Embarked (2 values). Also, looking at the Fare summary, we see that the minimum is 0. That mean either some people are getting in for free or those are also missing values.

Let's first convert everything to numerical values, so that it's much easier for us to work with. From the info description, we see that Name, Sex, Ticket, Cabin, and Embarked are all non-numeric values, so we will convert them using the following block of code:

`python
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

train.info()
`

IMAGE

#### Feature Engineering

Now for the feature engineering. We will be creating three new features. The first is family size. We know the count of spouses and siblings, and we also know the count of parents and grandparents. Combining these two we can get the count of how big each person's family is which might be useful. We will then use this family size to judge whether they came in single or in a small family or a big family.

`python
train['Family'] = train.SibSp + train.Parch + 1
train['Child'] = train.Age.apply(lambda x: 1 if x < 18 else 0)
train['Single'] = train.Family.apply(lambda x: 1 if x == 1 else 0)
train['SmallFam'] = train.Family.apply(lambda x: 1 if x < 4 else 0)
train['LargeFam'] = train.Family.apply(lambda x: 1 if x >= 4 else 0)
`

Awesome. Now lets look at the Cabin and Ticket features. The Cabin seems really sparse compared to the others, but the ticket feature seems really interesting! There are ticket numbers that overlap and looking even closer we can see that a few of the consecutive tickets are even from the same family. Lets play with this to see if we can get anything useful from it. We first loop through all the rows that are sorted based on just the ticket number (note that some tickets have something in front of the numbers and some say "LINE" so we will omit this for the purpose of this feature). We separate tickets into different groups if they are consecutive and then we will count how many of them are in the same groups. This created a total of 432 groups which can help us mix and match beyond family, such as nannies and such.

`python
def ticket_sort(x):
    last = 0
    n = 0
    x['Group'] = 0
    x_sort = x.sort_values('Ticket')
    for i, row in x_sort.iterrows():
        if int(row['Ticket']) - last == 0:
            x_sort.loc[i, 'Group'] = n
        elif int(row['Ticket']) - last == 1:
            x_sort.loc[i, 'Group'] = n
            last = int(row['Ticket'])
        else:
            n += 1
            last = int(row['Ticket'])
            x_sort.loc[i, 'Group'] = n
    return x_sort

train = ticket_sort(train)
group_count = train.groupby("Group").count()
train['GroupSize'] = train.Group.apply(lambda x: group_count['PassengerId'][x])
`

#### Dealing with Missing Values
