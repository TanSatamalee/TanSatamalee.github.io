---
layout: post
title: Titanic Kaggle Analysis
---

Hello to who ever is reading this! This will be my first blog ever so be warned! For my first, I chose to work with the [Titanic Kaggle Dataset](https://www.kaggle.com/c/titanic) and do a straigthforward analysis with random forests after cleaning and processing it.

We first load in the two datasets we are going to be working with.

```python
# Loading Training and Test Datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
```

After importing the data, we first take a glance and look at what we are working with.

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
# Converts Embarked categorical variables to numeric ordered by departure from first to last port.
def convert_embarked(x):
    if x == 'S':
        return 2
    if x == 'C':
        return 1
    if x == 'Q':
        return 3

# Converts sex to numeric.
def convert_sex(x):
    if x == 'male':
        return 1
    if x == 'female':
        return 0

# Converts ticket to numeric by eliminating any predecessing letters in front of the ticket number.
def convert_ticket(x):
    x = x.split()[-1]
    if x.isnumeric():
        return int(x)
    return -1

# Converts cabins into numerical values using just the first letter of the cabin (omits the letter portion).
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
    
# Converts the title into numerics and combines titles taht seem similar enough to each other.
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

# Applies the converting functions to Embarked, Sex, Ticket Number, and Cabin.
train.Embarked = train.Embarked.apply(convert_embarked)
train.Sex = train.Sex.apply(convert_sex)
train.Ticket = train.Ticket.apply(convert_ticket)
train.Cabin.fillna('None', inplace=True)
train.Cabin = train.Cabin.apply(convert_cabin)

train.info()
```

IMAGE

#### Feature Engineering

Now for the feature engineering. We will be creating three new features. The first is family size. We know the count of spouses and siblings, and we also know the count of parents and grandparents. Combining these two we can get the count of how big each person's family is which might be useful. We will then use this family size to judge whether they came in single or in a small family or a big family.

```python
# Crating new features.
train['Family'] = train.SibSp + train.Parch + 1 # Family includes siblings, spouses, parents, and self.
train['Child'] = train.Age.apply(lambda x: 1 if x < 18 else 0) # Child is anyone below 18.
train['Single'] = train.Family.apply(lambda x: 1 if x == 1 else 0) # People who are on the ship alone would only have a Family count of 1.
train['SmallFam'] = train.Family.apply(lambda x: 1 if x < 4 else 0) # Small family consists of four people.
train['LargeFam'] = train.Family.apply(lambda x: 1 if x >= 4 else 0) # Large family consists of more than four people.
```

Awesome. Now lets look at the Cabin and Ticket features. The Cabin seems really sparse compared to the others, but the ticket feature seems really interesting! There are ticket numbers that overlap and looking even closer we can see that a few of the consecutive tickets are even from the same family. Lets play with this to see if we can get anything useful from it. We first loop through all the rows that are sorted based on just the ticket number (note that some tickets have something in front of the numbers and some say "LINE" so we will omit this for the purpose of this feature). We separate tickets into different groups if they are consecutive and then we will count how many of them are in the same groups. This created a total of 432 groups which can help us mix and match beyond family, such as nannies and such.

```python
# Sorting the tickets by their numbers and assigning a group number to consecutive tickets.
def ticket_sort(x):
    last = 0 # Keeps track of the last number in the loop.
    n = 0 # Keeps track of the group number the loop is currently on.
    x['Group'] = 0 # Initializes Group column to be all 0s.
    x_sort = x.sort_values('Ticket')
    # Iterates through the rows and assigns the group number if the ticket numbers are consecutive, else assigns a new group number.
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

# Creates the grouping function on data and then counts how many people are in each group.
train = ticket_sort(train)
group_count = train.groupby("Group").count()
train['GroupSize'] = train.Group.apply(lambda x: group_count['PassengerId'][x])
```

Finally, lets look at the untouched Name category. It seems everyone has some sort of title that goes with their names so we will extract that.

```python
train['Title'] = train.Name.str.extract('\, ([A-Z][^ ]*\.)',expand=False)
train.Title = train.Title.apply(convert_title)
```

#### Dealing with Missing Values

Including our new features that were generated, we are missing values from Title, Embarked, Age, and Feats (since we are omitting Cabin due to sparsity). For Title and Embarked, we are just going to go with the main majority since they are missing only a few values and there is a clear majority. For Age and Fare, we are going to derive these based on the mean or median of other variables.

```python
# Fills the few missing Titles as Mr. since thats the most common.
train.Title = train.Title.fillna(5)
# Fills the Embarked with most common also.
train.Embarked = train.Embarked.fillna(2)
# Fills any Fare with a value of Zero with a NaN and then uses Family as a basis of filling the fare with the median of that Family.
train.Fare[train.Fare == 0] = np.nan
train.Fare = train.groupby("Family").transform(lambda x: x.fillna(x.median())).Fare
# Fills missing Age with value based on Title since the two seem to relate a lot. (Considering to try rpart model)
train.Age = train.groupby("Title").transform(lambda x: x.fillna(x.mean())).Age

# Looking at Age correlations with other features to see if Age can be predicted from others.
fig, axs = plt.subplots(2, 5)
fig.set_size_inches(20, 10)
train_plot = train.dropna()
sns.boxplot('Pclass', 'Age', data=train_plot, ax=axs[0][0])
sns.boxplot('Sex', 'Age', data=train_plot, ax=axs[0][1])
sns.boxplot('SibSp', 'Age', data=train_plot, ax=axs[0][2])
sns.boxplot('Parch', 'Age', data=train_plot, ax=axs[0][3])
sns.boxplot('Ticket', 'Age', data=train_plot, ax=axs[0][4])
sns.boxplot('Fare', 'Age', data=train_plot, ax=axs[1][0])
sns.boxplot('Cabin', 'Age', data=train_plot, ax=axs[1][1])
sns.boxplot('Embarked', 'Age', data=train_plot, ax=axs[1][2])
sns.boxplot('Title', 'Age', data=train_plot, ax=axs[1][3])
sns.boxplot('Family', 'Age', data=train_plot, ax=axs[1][4])

# Looking at Fare correlations with other features to see if Fare can be predicted from others.
fig, axs = plt.subplots(2, 5)
fig.set_size_inches(20, 10)
train_plot = train.dropna()[train.Fare < 300]
sns.boxplot('Pclass', 'Fare', data=train_plot, ax=axs[0][0])
sns.boxplot('Sex', 'Fare', data=train_plot, ax=axs[0][1])
sns.boxplot('SibSp', 'Fare', data=train_plot, ax=axs[0][2])
sns.boxplot('Parch', 'Fare', data=train_plot, ax=axs[0][3])
sns.boxplot('Ticket', 'Fare', data=train_plot, ax=axs[0][4])
sns.boxplot('Age', 'Fare', data=train_plot, ax=axs[1][0])
sns.boxplot('Cabin', 'Fare', data=train_plot, ax=axs[1][1])
sns.boxplot('Embarked', 'Fare', data=train_plot, ax=axs[1][2])
sns.boxplot('Title', 'Fare', data=train_plot, ax=axs[1][3])
sns.boxplot('Family', 'Fare', data=train_plot, ax=axs[1][4])
```

IMAGE

By graphing scatter plots of each of these versus the other features, we see that Title is a pretty good predictor for Age and Family a fairly good predictor for Fare, so we will use those to predict the missing Age and Fare.

#### Training the Classifier

For this classifier, we will be using the random forests algorithm. This is just because ensembel learning seems to be the best choice when compared to the other ones for this situation (We don't expect linear correlations between the variables so Logistic Regression might not be optimal). We tune each parameter one-by-one and pick the most optimal ones for our final prediction. We then utilize k-fold cross validation to check our predictor error and ended with a result of about 0.82 score.

```python
# Tuning the kfolds parameter. (Result 50)
print('kfolds hypertune')
kfolds = [5, 10, 20, 50, 70, 100]
for i in kfolds:
    tune_rf = RandomForestClassifier()
    print(np.mean(cross_val_score(tune_rf, X, Y, cv=i)))

# Tuning number of trees to use. (Result 50)
print('n_estimate hypertune')
trees = [1, 5, 10, 20, 50, 70, 100]
for i in trees:
    tune_rf = RandomForestClassifier(n_estimators=i)
    print(np.mean(cross_val_score(tune_rf, X, Y, cv=50)))

# Tuning maximum tree depth. (Result 5)
print('max_depth hypertune')
max_depth = [1, 5, 10, 20, 50, 100, 200]
for i in max_depth:
    tune_rf = RandomForestClassifier(n_estimators=50, max_depth=i)
    print(np.mean(cross_val_score(tune_rf, X, Y, cv=50)))

# Tuning minimum amount of samples for a tree split. (Result 0.001)
print('min_split hypertune')
min_split = [0.00001, 0.0001, 0.001, 0.01, 0.1]
for i in min_split:
    tune_rf = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_split=i)
    print(np.mean(cross_val_score(tune_rf, X, Y, cv=50)))

# Tuning minimum amount of samples to create a new leaf. (Result 0.0001)
print('min_leaf hypertune')
min_leaf = [0.00001, 0.0001, 0.001, 0.01, 0.1]
for i in min_leaf:
    tune_rf = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_split=0.001, min_samples_leaf=i)
    print(np.mean(cross_val_score(tune_rf, X, Y, cv=50)))

# Tuning maximum amount of features to consider when deciding on a split. (Result 2)
print('min_split hypertune')
max_feat = [1, 2, 3, 4]
for i in max_feat:
    tune_rf = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_split=0.001, min_samples_leaf=0.0001, max_features=i)
    print(np.mean(cross_val_score(tune_rf, X, Y, cv=50)))

# Creating final model after hypertuning and then testing the final result fo the cross vallidation.
rf = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_split=0.001, min_samples_leaf=0.0001, max_features=2)
X = train[train_feats]
Y = train['Survived']

print(np.mean(cross_val_score(rf, X, Y, cv=20)))
```
