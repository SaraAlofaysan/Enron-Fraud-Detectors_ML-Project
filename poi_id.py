
# coding: utf-8

# In[1]:

import sys
import pickle
import pandas as pd
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit

from tester import dump_classifier_and_data
from sklearn.feature_selection import SelectKBest

from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn import linear_model
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import tree

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score


# In[ ]:




# ## Feature Selection and Data Loading

# In[2]:

### Task 1: Select what features you'll use.
### The first feature must be "poi".
## I will study all features then i will elimenate the un needed ones
features_list = ['poi','salary','bonus','to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi','shared_receipt_with_poi', 'deferral_payments', 'total_payments', 'loan_advances','restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive','restricted_stock','director_fees']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# ## Remove Outliers

# In[3]:

## ---- After exploring the data dictionary, I found the three outliers and I decided to eleimenate them as follows:
# Data Dictionary Key ' Total'
data_dict.pop("TOTAL")
# Data Dictionary Key ' THE TRAVEL AGENCY IN THE PARK'
data_dict.pop("THE TRAVEL AGENCY IN THE PARK")
# Data Dictionary Key ' LOCKHART EUGENE E' has NAN valuse for all features
data_dict.pop("LOCKHART EUGENE E")


# ## Create new feature(s)

# In[4]:

### Task 3: Create new feature(s)

## First, I will transform data dictionary to dataframe 

df = pd.DataFrame.from_dict(data_dict,orient='index')
df = df.loc[:,features_list]
df = df.replace('NaN', np.nan)

## Then, I will create new feature(s)
# Bonus-salary ratio
df['ratio_bonus_salary'] = df['bonus']/df['salary']
# from_this_person_to_poi
df['ratio_to_poi'] = df['from_this_person_to_poi']/df['to_messages']
# from_poi_to_this_person
df['ratio_from_poi'] = df['from_poi_to_this_person']/df['from_messages']
# ratio of poi msgs of all msgs
df['ratio_poi_message'] = df['shared_receipt_with_poi']/(df['to_messages']+df['from_messages'])

## change NANs to zero for better calculations
df = df.replace(np.nan, 0.)
## change poi to bolan 0 or 1
df['poi'] = df['poi']*1.0


# In[5]:

df.head(5)


# In[6]:

### Feature Selection ###
# I droped the un-needed features as the below features will not provide extra informtion after i reengenerieed a new one

drop1_key =[
'to_messages',
'from_messages',
'shared_receipt_with_poi',
'from_this_person_to_poi',
'from_poi_to_this_person',
'email_address'
]
df.drop(drop1_key, axis=1,inplace = True)


# ## Test and Select the k-Best Features

# In[7]:

#### K-best algorithm: use statistical inference to see the importance of each feature
def run_K_best(df,target,number):
    
    features = list(df.columns.values)
    features.remove(target)
    features_name = [target] + features
    
    k_best = SelectKBest(k=number)
    k_best.fit(df[features].as_matrix(), df[target].as_matrix())
    score = k_best.scores_
    score_chart = zip(features, score)
    score_chart_df = pd.DataFrame(score_chart, columns=['Feature', 'Score'])
    return score_chart_df.sort_values('Score', axis=0,ascending=False)

print run_K_best(df,'poi','all')


# In[8]:

### Feature Selection ###
# I droped the low score features from the dictionary
drop1_key =[
'deferral_payments',
'restricted_stock_deferred',
'director_fees',
]
df.drop(drop1_key, axis=1,inplace = True)


# ### Define the new feature list after the feature enginering and selection

# In[9]:

column_name = list(df.columns.values)
features_list= list(df.columns.values)
column_name.remove('poi')


# ### Transform dataframe back to dictionary and store it to my_dataset

# In[10]:

my_dataset = df.to_dict(orient="index")


# ## Extract features and labels from dataset for local testing

# In[11]:

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# ## Build a varity of classifiers for further testing

# In[12]:

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.

######## 1) GaussianNB #######
g_clf = GaussianNB()

########  2) Support Vector Machine Classifier ########
s_clf = svm.SVC(kernel='rbf', C=1000,gamma = 0.0001,random_state = 42, class_weight = 'auto')

#######   3) Decision Tree   #######
d_clf = tree.DecisionTreeClassifier(min_samples_split=101)

######    4)   K-means Clustering   #######
k_clf = KMeans(n_clusters=2, tol=0.001)

######   5)  Random Forest    #######
rf_clf = RandomForestClassifier(max_depth = 5,max_features = 'sqrt',n_estimators = 10, random_state = 42)

######   6)  AdaBoost    #######
ad_clf = AdaBoostClassifier()


# In[13]:

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method.

def evaluate_clf(clf, features, labels, num_iters=500):
    
    accuracy = []
    precision = []
    recall = []

    first = True
    for trial in range(num_iters):
        features_train, features_test, labels_train, labels_test =train_test_split(features, labels, test_size=0.3,random_state=42)
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        accuracy.append(accuracy_score(labels_test, predictions))
        precision.append(precision_score(labels_test, predictions))
        recall.append(recall_score(labels_test, predictions))
        
        if trial % 10 == 0:
            if first:
                sys.stdout.write('\nProcessing')
            sys.stdout.write('.')
            sys.stdout.flush()
            first = False

    print clf
    print "accuracy: {}".format(np.mean(accuracy))
    print "precision: {}".format(np.mean(precision))
    print "recall:    {}".format(np.mean(recall))
    return np.mean(accuracy),np.mean(precision), np.mean(recall)


# In[14]:

### Evaluate all clasifiers
evaluate_clf(g_clf, features, labels)
evaluate_clf(s_clf, features, labels)
evaluate_clf(d_clf, features, labels)
evaluate_clf(k_clf, features, labels)
evaluate_clf(rf_clf, features, labels)
evaluate_clf(ad_clf, features, labels)


# In[15]:

### Select Gaussian NB as final algorithm
clf = g_clf


# In[16]:

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)


# In[17]:




# In[ ]:



