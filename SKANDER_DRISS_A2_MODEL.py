#!/usr/bin/env python
# coding: utf-8

# <h1>Machine Learning Assignment 2</h1>
# <h2>Submitted by Skander Driss</h2>
# <h4>Professor Chase Kusterer<br>
# Hult International Business School</h4>

# In[5]:


get_ipython().run_line_magic('timeit', '')
# importing libraries

import pandas as pd                                  # data science essentials
import random            as rand                     # random number gen
import matplotlib.pyplot as plt                      # essential graphical output
import seaborn as sns                                # enhanced graphical output
from sklearn.model_selection import train_test_split # train-test split
from sklearn.linear_model import LogisticRegression  # logistic regression
from sklearn.metrics import confusion_matrix         # confusion matrix
from sklearn.metrics import roc_auc_score            # auc score
from sklearn.neighbors import KNeighborsClassifier   # KNN for classification
from sklearn.neighbors import KNeighborsRegressor    # KNN for regression
from sklearn.preprocessing import StandardScaler     # standard scaler
import statsmodels.formula.api as smf
from sklearn.model_selection import GridSearchCV     # hyperparameter tuning
from sklearn.metrics import make_scorer              # customizable scorer
from IPython.display import Image                    # displays on frontend
import pydotplus
from sklearn.tree import DecisionTreeClassifier
# interprets dot objects

get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')

# setting pandas print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# specifying file name
file = 'Apprentice_Chef_Dataset.xlsx'


# reading the file into Python
food = pd.read_excel(file)

# Imputing the missing values: Family name missing will take the First name
food['FAMILY_NAME'][food['FAMILY_NAME'].isnull()] = food['FIRST_NAME'][food['FAMILY_NAME'].isnull()]


############################ Feature Engineering

# Emails

# placeholder list
placeholder_lst = []

# looping over each email address
for index, col in food.iterrows():
    
    # splitting email domain at '@'
    split_email = food.loc[index, 'EMAIL'].split(sep = '@')
    
    # appending placeholder_lst with the results
    placeholder_lst.append(split_email)
    

# converting placeholder_lst into a DataFrame 
email_df = pd.DataFrame(placeholder_lst)


# STEP 2: concatenating with original DataFrame

# renaming column to concatenate
email_df.columns = ['name' ,'email_domain']


# concatenating personal_email_domain with friends DataFrame
food = pd.concat([food, email_df['email_domain']],
                   axis = 1)


# personal and junk email domain
personal_email_domains = ['@gmail.com', '@yahoo.com','@protonmail.com']
junk_email_domains=['@me.com','@aol.com' ,'@hotmail.com' ,'@live.com' ,'@msn.com' ,'@passport.com']

# placeholder list
placeholder_lst = []

# looping to group observations by domain type
for domain in food['email_domain']:
        if '@' + domain in personal_email_domains :
            placeholder_lst.append('Personal_email')
        elif '@' + domain in junk_email_domains :
            placeholder_lst.append('Junk_email')
        else:
            placeholder_lst.append('Professional_email')


# concatenating with original DataFrame
food['domain_group'] = pd.Series(placeholder_lst )

food['domain_group'].astype('category')

#hot coding for domain (Professional/Personal)
one_hot_domain = pd.get_dummies(food['domain_group'])
food = food.join([one_hot_domain])


############Prepation for Treshold

unique_meals_change_hi         = 12.5
CONTACTS_W_CUSTOMER_SERVICE_lo = 5
CONTACTS_W_CUSTOMER_SERVICE_hi = 8
cancellation_before_noon_hi    = 5
cancellation_after_noon_hi     = 2
pc_logins_lo                   = 4
pc_logins_hi                   = 7
MOBILE_LOGINS_lo               = 0
MOBILE_LOGINS_hi               = 3
MASTER_CLASSES_ATTENDED_hi     = 2
AVG_CLICKS_PER_VISIT_lo        = 7

cancellation_after_noon_change_at = 0 # zero inflated
weekly_plan_change_at             = 0 # zero inflated
total_photo_viewed_change_at      = 0 # zero inflated

# UNIQUE_MEALS_PURCH
food['change_unique_meals'] = 0
condition = food.loc[0:,'change_unique_meals'][food['UNIQUE_MEALS_PURCH'] > unique_meals_change_hi]

food['change_unique_meals'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

# Contact with customer service
food['out_CONTACTS_W_CUSTOMER_SERVICE'] = 0
condition_hi = food.loc[0:,'out_CONTACTS_W_CUSTOMER_SERVICE'][food['CONTACTS_W_CUSTOMER_SERVICE'] >= CONTACTS_W_CUSTOMER_SERVICE_hi]
condition_lo = food.loc[0:,'out_CONTACTS_W_CUSTOMER_SERVICE'][food['CONTACTS_W_CUSTOMER_SERVICE'] <= CONTACTS_W_CUSTOMER_SERVICE_lo]

food['out_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

food['out_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

#cancellation_before_noon
food['out_cancellation_before_noon'] = 0
condition_hi = food.loc[0:,'out_cancellation_before_noon'][food['CANCELLATIONS_BEFORE_NOON'] > cancellation_before_noon_hi ]

food['out_cancellation_before_noon'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

#cancellation_after_noon
food['out_cancellation_after_noon'] = 0
condition_hi = food.loc[0:,'out_cancellation_after_noon'][food['CANCELLATIONS_AFTER_NOON'] > cancellation_after_noon_hi ]

food['out_cancellation_after_noon'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

# Pc logins
food['out_PC_LOGINS'] = 0
condition_hi = food.loc[0:,'out_PC_LOGINS'][food['PC_LOGINS'] >= pc_logins_hi]
condition_lo = food.loc[0:,'out_PC_LOGINS'][food['PC_LOGINS'] <= pc_logins_lo]

food['out_PC_LOGINS'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

food['out_PC_LOGINS'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

# mobile logins
food['out_MOBILE_LOGINS'] = 0
condition_hi = food.loc[0:,'out_MOBILE_LOGINS'][food['MOBILE_LOGINS'] >= MOBILE_LOGINS_hi ]
condition_lo = food.loc[0:,'out_MOBILE_LOGINS'][food['MOBILE_LOGINS'] <= MOBILE_LOGINS_lo]
food['out_MOBILE_LOGINS'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

food['out_MOBILE_LOGINS'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

#MASTER_CLASSES_ATTENDED
food['out_MASTER_CLASSES_ATTENDED'] = 0
condition_hi = food.loc[0:,'out_MASTER_CLASSES_ATTENDED'][food['MASTER_CLASSES_ATTENDED'] >= MASTER_CLASSES_ATTENDED_hi]

food['out_MASTER_CLASSES_ATTENDED'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

#avg click per visit
food['out_AVG_CLICKS_PER_VISIT'] = 0
condition_hi = food.loc[0:,'out_AVG_CLICKS_PER_VISIT'][food['AVG_CLICKS_PER_VISIT'] > AVG_CLICKS_PER_VISIT_lo]

food['out_AVG_CLICKS_PER_VISIT'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

# CANCELLATIONS_AFTER_NOON
food['change_cancellation_after_noon'] = 0
condition = food.loc[0:,'change_cancellation_after_noon'][food['CANCELLATIONS_AFTER_NOON'] == cancellation_after_noon_change_at]

food['change_cancellation_after_noon'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)

# WEEKLY_PLAN
food['change_weekly_plan'] = 0
condition = food.loc[0:,'change_weekly_plan'][food['WEEKLY_PLAN'] == weekly_plan_change_at]

food['change_weekly_plan'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)

# Overall Cond
food['change_total_photo_viewed'] = 0
condition = food.loc[0:,'change_total_photo_viewed'][food['TOTAL_PHOTOS_VIEWED'] == total_photo_viewed_change_at]

food['change_total_photo_viewed'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)

# placeholder list for PRODUCT_CATEGORIES_VIEWED
placeholder_lst = []

# looping to group observations by category viewed
for i in food['PRODUCT_CATEGORIES_VIEWED']:
        if i == 1:
            placeholder_lst.append('Categ_Viewed_1')
        elif i == 2:
            placeholder_lst.append('Categ_Viewed_2')
        elif i == 3:
            placeholder_lst.append('Categ_Viewed_3')
        elif i == 4:
            placeholder_lst.append('Categ_Viewed_4')
        elif i == 5:
            placeholder_lst.append('Categ_Viewed_5')
        elif i == 6:
            placeholder_lst.append('Categ_Viewed_6')
        elif i == 7:
            placeholder_lst.append('Categ_Viewed_7')
        elif i == 8:
            placeholder_lst.append('Categ_Viewed_8')
        elif i == 9:
            placeholder_lst.append('Categ_Viewed_9')
        else:
            placeholder_lst.append('Categ_Viewed_10')

# concatenating with original DataFrame
food['categ_viewed'] = pd.Series(placeholder_lst )

food['categ_viewed'].astype('category')

#hot coding
one_hot_categ_viewed = pd.get_dummies(food['categ_viewed'])
food = food.join([one_hot_categ_viewed])


# placeholder list for FOLLOWED_RECOMMENDATIONS_PCT
placeholder_lst = []

# looping to group observations by Followed Recommendations
for i in food['FOLLOWED_RECOMMENDATIONS_PCT']:
        if i == 0:
            placeholder_lst.append('Followed_rec_0')
        elif i == 10:
            placeholder_lst.append('Followed_rec_10')
        elif i == 20:
            placeholder_lst.append('Followed_rec_20')
        elif i == 30:
            placeholder_lst.append('Followed_rec_30')
        elif i == 40:
            placeholder_lst.append('Followed_rec_40')
        elif i == 50:
            placeholder_lst.append('Followed_rec_50')
        elif i == 60:
            placeholder_lst.append('Followed_rec_60')
        elif i == 70:
            placeholder_lst.append('Followed_rec_70')
        elif i == 80:
            placeholder_lst.append('Followed_rec_80')
        else:
            placeholder_lst.append('Followed_rec_90')


# concatenating with original DataFrame
food['Followed_rec'] = pd.Series(placeholder_lst )

food['Followed_rec'].astype('category')

#hot coding 
one_hot_Followed_rec = pd.get_dummies(food['Followed_rec'])
food = food.join([one_hot_Followed_rec])


# placeholder list for MEDIAN_MEAL_RATING
placeholder_lst = []

# looping to group observations by Rating
for i in food['MEDIAN_MEAL_RATING']:
        if i == 1:
            placeholder_lst.append('1_star_rating')
        elif i == 2:
            placeholder_lst.append('2_star_rating')
        elif i == 3:
            placeholder_lst.append('3_star_rating')
        elif i == 4:
            placeholder_lst.append('4_star_rating')
        elif i == 5:
            placeholder_lst.append('5_star_rating')
            
# concatenating with original DataFrame
food['Star_rating'] = pd.Series(placeholder_lst )

food['Star_rating'].astype('category')

#hot coding 
one_hot_Star_rating = pd.get_dummies(food['Star_rating'])
food = food.join([one_hot_Star_rating])

# declaring explanatory variables and dropping the invalid columns
food_data = food.drop(['CROSS_SELL_SUCCESS',
                      'NAME',
                      'EMAIL', 
                      'FIRST_NAME', 
                      'FAMILY_NAME', 
                      'email_domain', 
                      'domain_group',
                      'categ_viewed',
                      'Followed_rec',
                      'Star_rating'],
                      axis = 1)


# declaring response variable
food_target = food.loc[ : , 'CROSS_SELL_SUCCESS']

# train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
            food_data,
            food_target,
            test_size = 0.25,
            random_state = 508,
            stratify = food_target)


# merging training data for statsmodels
food_train = pd.concat([X_train, y_train], axis = 1)


###check it
# instantiating a logistic regression model object
logistic_full = smf.logit(formula = """ CROSS_SELL_SUCCESS ~
                                        MOBILE_NUMBER +
                                        TASTES_AND_PREFERENCES +
                                        FOLLOWED_RECOMMENDATIONS_PCT +
                                        AVG_PREP_VID_TIME +
                                        TOTAL_PHOTOS_VIEWED +
                                        out_CONTACTS_W_CUSTOMER_SERVICE +
                                        out_cancellation_before_noon +
                                        change_cancellation_after_noon +
                                        change_weekly_plan +
                                        change_total_photo_viewed """,
                                        data    = food_train)


# fitting the model object
results_full = logistic_full.fit()


# checking the results SUMMARY
results_full.summary()


# declaring a hyperparameter space
criterion_space = ['gini', 'entropy']
splitter_space = ['best', 'random']
depth_space = pd.np.arange(1, 25)
leaf_space  = pd.np.arange(1, 100)


# creating a hyperparameter grid
param_grid = {'criterion'        : criterion_space,
              'splitter'         : splitter_space,
              'max_depth'        : depth_space,
              'min_samples_leaf' : leaf_space}


# INSTANTIATING the model object without hyperparameters
tuned_tree = DecisionTreeClassifier(random_state = 802)


# GridSearchCV object
tuned_tree_cv = GridSearchCV(estimator  = tuned_tree,
                             param_grid = param_grid,
                             cv         = 3,
                             scoring    = make_scorer(roc_auc_score,
                                                      needs_threshold = False))


# FITTING to the FULL DATASET (due to cross-validation)
tuned_tree_cv.fit(food_data, food_target)


# PREDICT step is not needed


# printing the optimal parameters and best score
print("Tuned Parameters  :", tuned_tree_cv.best_params_)
print("Tuned Training AUC:", tuned_tree_cv.best_score_.round(4))

# building a model based on hyperparameter tuning results

# INSTANTIATING a logistic regression model with tuned values
tree_tuned = tuned_tree_cv.best_estimator_


# FIT step is not needed


# PREDICTING based on the testing set
tree_tuned_pred = tree_tuned.predict(X_test)


# SCORING the results
print('Training ACCURACY:', tree_tuned.score(X_train, y_train).round(4))
print('Testing  ACCURACY:', tree_tuned.score(X_test, y_test).round(4))
print('AUC Score        :', roc_auc_score(y_true  = y_test,
                                          y_score = tree_tuned_pred).round(4))


# In[ ]:




