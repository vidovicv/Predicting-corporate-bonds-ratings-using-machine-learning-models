#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import math
from numpy import mean
from numpy import std
from numpy import absolute
from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import Lasso
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from statistics import mean, stdev
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay




dataset = pd.read_excel("data.xlsx")
headers = dataset.iloc[0]
bonds_dataset = pd.DataFrame(dataset.values[1:], columns=headers)


# In[ ]:


bonds_dataset.drop(bonds_dataset[bonds_dataset['CUR_MKT_CAP']== '#N/A Invalid Security'].index, axis=0, inplace=True)
bonds_dataset.dropna(subset = ["CRNCY"], inplace=True)
bonds_dataset.drop(bonds_dataset[bonds_dataset['INDUSTRY_SECTOR']== 'Government'].index, axis=0, inplace=True)
bonds_dataset.reset_index(inplace=True)
bonds_dataset = bonds_dataset.replace(["#N/A Field Not Applicable"],np.nan)


bonds_dataset["RTG_MOODY"].replace({"Aa2u": "Aa2",
                                    "Aaau": "Aaa",
                                    "Baa3u": "Baa3",
                                    "Aa3u": "Aa3",
                                    "A3u": "A3",
                                    "Baa2u": "Baa2",
                                    "A3 *-": "A3",
                                    "Aa1u": "Aa1",
                                    "Ba2u": "Ba2",
                                    "Baa3 *-": "Baa3",
                                    "Baa1u": "Baa1",
                                    "A1u": "A1",
                                    "Ba1u": "Ba1",
                                    "Baa3 *+": "Baa3",
                                    "Ba1 *+": "Ba1",
                                    "Aa2 *+": "Aa2",
                                    "Aa3 *-": "Aa3",
                                    "Caa2u": "Caa2",
                                    "A3 *+": "A3",
                                    "Ba3u": "Ba3"}, inplace=True)

bonds_dataset["RTG_FITCH"].replace({"AAu": "AA",
                                    "AAAu": "AAA",
                                    "BBBu": "BBB",
                                    "AA-u": "AA-",
                                    "BBB+u": "BBB+",
                                    "A+u": "A+",
                                    "BBB+ *-": "BBB+",
                                    "Au": "A",
                                    "AA+u": "AA+",
                                    "BBB- *-": "BBB-",
                                    "A-u": "A-",
                                    "BB-u": "BB-",
                                    "AA- *-": "AA-",
                                    "BBB+ *+": "BBB+",
                                    "A+u(EXP)": "A+",
                                    "PIF": "C",
                                    "A *-": "A",
                                    "BBB-u": "BBB-",
                                    "B- *-": "B-"}, inplace=True) 

bonds_dataset["RTG_SP"].replace({"BBB+ *-": "BBB+",
                                    "BB- *-": "BB-",
                                    "BBB- *+": "BBB-",
                                    "BB- *-": "BB-",
                                    "BBB- *+": "BBB-",
                                    "A+u": "A+",
                                    "A- *-": "A-",
                                    "A+ *-": "A+",
                                    "(P)A-": "A-",
                                    "(P)AA-": "AA-",
                                    "B- *-": "B-",
                                    "BB+p": "BB+",
                                    "(P)AAA": "AAA",
                                    "AAu": "AA",
                                    "A+u(EXP)": "A+",
                                    "B+ *+": "B+",
                                    "A *-": "A"}, inplace=True)


bonds_dataset.loc[bonds_dataset['CNTRY_OF_RISK'].isna(),'CNTRY_OF_RISK'] = bonds_dataset.loc[bonds_dataset['CNTRY_OF_RISK'].isna(),'CNTRY_OF_DOMICILE']
bonds_dataset.loc[bonds_dataset['CNTRY_OF_DOMICILE'].isna(),'CNTRY_OF_DOMICILE'] = bonds_dataset.loc[bonds_dataset['CNTRY_OF_DOMICILE'].isna(),'CNTRY_OF_RISK']


NUMERICAL_COLUMNS = [19,20, 22,23,26,27,28]
                   
num_imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
num_imputer = num_imputer.fit(bonds_dataset.iloc[:,NUMERICAL_COLUMNS])                             
bonds_dataset.iloc[:,NUMERICAL_COLUMNS] = num_imputer.transform(bonds_dataset.iloc[:,NUMERICAL_COLUMNS])
bonds_dataset.drop(bonds_dataset[bonds_dataset['PUTABLE'].isna()].index, inplace=True)
pd.set_option('display.max_columns', None)

# take only bonds with at least one grade
with_rating = bonds_dataset[(bonds_dataset['RTG_MOODY'].notna())| (bonds_dataset['RTG_FITCH'].notna())| (bonds_dataset['RTG_SP'].notna())]

x_ = with_rating[['RISK_PREMIUM',
                               'CPN',                     'CPN_TYP',
                         'MTY_YEARS',                 'DUR_ADJ_MID',
                          'CALLABLE',                     'PUTABLE',
                       'YLD_YTM_MID',                 'YLD_CNV_MID',
                      'PX_DIRTY_MID',                  'REDEMP_VAL',
                   'INDUSTRY_SECTOR',              'INDUSTRY_GROUP',
                     'CNTRY_OF_RISK',           'CNTRY_OF_DOMICILE']]   
inputs = pd.get_dummies(x_, dtype=float)
names = inputs.columns


# In[ ]:


# make output numerical values

with_rating_output = with_rating[['RTG_MOODY', 'RTG_FITCH', 'RTG_SP']]
with_rating_output.reset_index(inplace=True)
with_rating_output['Grade'] = np.nan

investment_grade = ['Aaa', 'Aa1', 'Aa2','Aa3', 'A1', 'A2', 'A3', 'Baa1','Baa2','Baa3', 'AAA', 'BBB+', 'A-','A', 'A+', 'BBB', 'BBB+', 'BBB-','AA', 'AA+', 'AA-']
middle_grade =[ 'Ba1' , 'Ba2', 'Ba3', 'B1', 'B2', 'B3', 'BB', 'BB+', 'BB-', 'B', 'B+', 'B-', 'CCC', 'CC', 'C' ]
junk_bond = ['WR','NR','C', 'Caa1', 'Caa3', 'Caa2' 'Ca', 'D', 'WD', 'CCC+', 'Ca', 'Caa2' ]


# grades = ['1', '0', '-1']

def set_grade_m(row):  
    grade = 0
    
    if (with_rating_output.loc[row, 'RTG_MOODY'] in investment_grade):
        grade = 1
    elif (with_rating_output.loc[row, 'RTG_MOODY'] in middle_grade):
        grade = 0
    elif (with_rating_output.loc[row, 'RTG_MOODY'] in junk_bond):
        grade = -1
    else:
        grade = -2
    
    return grade

def set_grade_f(row):
    grade = 0
    
    if (with_rating_output.loc[row, 'RTG_FITCH'] in investment_grade):
        grade = 1
    elif (with_rating_output.loc[row, 'RTG_FITCH'] in middle_grade):
        grade = 0
    elif (with_rating_output.loc[row, 'RTG_FITCH'] in junk_bond):
        grade = -1
    else:
        grade = -2
    
    return grade

def set_grade_s(row):
    grade = 0
    
    if (with_rating_output.loc[row, 'RTG_SP'] in investment_grade):
        grade = 1
    elif (with_rating_output.loc[row, 'RTG_SP'] in middle_grade):
        grade = 0
    elif (with_rating_output.loc[row, 'RTG_SP'] in junk_bond):
        grade = -1
    else:
        grade = -2
    
    return grade

for i in range(0, len(with_rating_output)):
    with_rating_output.loc[i, 'Grade'] = set_grade_m(i)
    

for i in range(0, len(with_rating_output)):
    if with_rating_output.loc[i,'Grade'] == -2:
        with_rating_output.loc[i, 'Grade' ] = set_grade_f(i)
        
for i in range(0, len(with_rating_output)):
    if with_rating_output.loc[i,'Grade'] == -2:
        with_rating_output.loc[i, 'Grade' ] = set_grade_s(i)


# In[ ]:


X = inputs.values
Y = with_rating_output['Grade'].values


# In[ ]:


pd.DataFrame(Y).value_counts()


# In[ ]:


pd.DataFrame(X)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=1, stratify=Y)


# ## Non-financial group

# ### model I: Balanced Bagging Classifier

# In[ ]:


from imblearn.ensemble import BalancedBaggingClassifier

classifier = DecisionTreeClassifier()
  
num_trees = 500
  
model_1 = BalancedBaggingClassifier(base_estimator = classifier,
                          n_estimators = num_trees)

model_1.fit(X_train,y_train)


# In[ ]:


cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)

scores = cross_val_score(model_1, X, Y, cv=cv, n_jobs=-1)

print(np.mean(scores))


# In[ ]:


import matplotlib.pyplot as plt

titles_options = [
    ("Confusion matrix, without normalization", None),
    ("Normalized confusion matrix", "true"),
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        model_1,
        X_test,
        y_test,
        display_labels=model_1.classes_,
        cmap=plt.cm.Blues,
        normalize=normalize,
    )
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()


# In[ ]:


from sklearn.metrics import classification_report

y_true = y_test
y_pred = model_1.predict(X_test)
target_names = ['-1.', '0.', '1.']

print(classification_report(y_true, y_pred, target_names=target_names))


# ### model II: Balanced Random Forest

# In[ ]:


from imblearn.ensemble import BalancedRandomForestClassifier


# In[ ]:


model_2 = BalancedRandomForestClassifier(n_estimators=300)


cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)

scores = cross_val_score(model_2, X, Y, cv=cv, n_jobs=-1)

print(np.mean(scores))


# In[ ]:


model_2.fit(X_train, y_train)


# In[ ]:


# from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

titles_options = [
    ("Confusion matrix, without normalization", None),
    ("Normalized confusion matrix", "true"),
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        model_2,
        X_test,
        y_test,
        display_labels=model_2.classes_,
        cmap=plt.cm.Blues,
        normalize=normalize,
    )
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()


# In[ ]:


from sklearn.metrics import classification_report

y_true = y_test
y_pred_2 = model_2.predict(X_test)
target_names = ['-1.', '0.', '1.']

print(classification_report(y_true, y_pred_2, target_names=target_names))


# ## Financial group

# In[ ]:


NUMERICAL_COLUMNS = [4,6,7,8,9,14,15,16,18,19]
                   
num_imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
num_imputer = num_imputer.fit(with_rating.iloc[:,NUMERICAL_COLUMNS])                             
with_rating.iloc[:,NUMERICAL_COLUMNS] = num_imputer.transform(with_rating.iloc[:,NUMERICAL_COLUMNS])


# In[ ]:


numerical_with_rating = with_rating[[
                    'SALES_REV_TURN',                 'CUR_MKT_CAP',
                        'NET_INCOME',            'GEO_GROW_NET_INC',
                'NET_DEBT_%_CAPITAL',          'WACC_TOTAL_CAPITAL',
                'TOT_DEBT_TO_EBITDA',          'TOTAL_PAYOUT_RATIO', 
                          'PE_RATIO',
             'TOT_DEBT_TO_TOT_ASSET',                'BS_TOT_ASSET',
             'RSK_BB_EQTY_PRICE_VOL',             'RETURN_ON_ASSET',
             'RETURN_ON_INV_CAPITAL']]

# for missing values create a dummie, replace NaN with 0 
numerical_with_rating["CUR_MKT_CAP_val"] = np.where(numerical_with_rating['CUR_MKT_CAP'].isna() == True, 0,numerical_with_rating['CUR_MKT_CAP'])
numerical_with_rating["CUR_MKT_CAP_is_missing"] = np.where(numerical_with_rating['CUR_MKT_CAP'].isna() == True, 1,0)


numerical_with_rating["TOT_DEBT_TO_EBITDA_val"] = np.where(numerical_with_rating['TOT_DEBT_TO_EBITDA'].isna() == True, 0,numerical_with_rating['TOT_DEBT_TO_EBITDA'])
numerical_with_rating["TOT_DEBT_TO_EBITDA_is_missing"] = np.where(numerical_with_rating['TOT_DEBT_TO_EBITDA'].isna() == True, 1,0)



numerical_with_rating["TOTAL_PAYOUT_RATIO_val"] = np.where(numerical_with_rating['TOTAL_PAYOUT_RATIO'].isna() == True, 0,numerical_with_rating['TOTAL_PAYOUT_RATIO'])
numerical_with_rating["TOTAL_PAYOUT_RATIO_is_missing"] = np.where(numerical_with_rating['TOTAL_PAYOUT_RATIO'].isna() == True, 1,0)


numerical_with_rating["PE_RATIO_val"] = np.where(numerical_with_rating['PE_RATIO'].isna() == True, 0,numerical_with_rating['PE_RATIO'])
numerical_with_rating["PE_RATIO_is_missing"] = np.where(numerical_with_rating['PE_RATIO'].isna() == True, 1,0)


numerical_with_rating["RETURN_ON_ASSET_val"] = np.where(numerical_with_rating['RETURN_ON_ASSET'].isna() == True, 0,numerical_with_rating['RETURN_ON_ASSET'])
numerical_with_rating["RETURN_ON_ASSET_is_missing"] = np.where(numerical_with_rating['RETURN_ON_ASSET'].isna() == True, 1,0)


# In[ ]:


X_2 = numerical_with_rating[['SALES_REV_TURN', 'NET_INCOME', 'GEO_GROW_NET_INC',
       'NET_DEBT_%_CAPITAL', 'WACC_TOTAL_CAPITAL',
       'TOT_DEBT_TO_TOT_ASSET',
       'BS_TOT_ASSET', 'RSK_BB_EQTY_PRICE_VOL',
       'RETURN_ON_INV_CAPITAL', 'CUR_MKT_CAP_val', 'TOT_DEBT_TO_EBITDA_val',
       'TOTAL_PAYOUT_RATIO_val', 'PE_RATIO_val', 'RETURN_ON_ASSET_val',
       'RETURN_ON_ASSET_val', 'CUR_MKT_CAP_is_missing',
       'TOT_DEBT_TO_EBITDA_is_missing', 'TOTAL_PAYOUT_RATIO_is_missing',
       'PE_RATIO_is_missing', 'RETURN_ON_ASSET_is_missing']]

list_numerical = X_2.columns
Y_2 = with_rating_output['Grade']


# In[ ]:


# from sklearn.preprocessing import StandardScaler
# data = X_2

# cols_to_scale = ['SALES_REV_TURN', 'NET_INCOME', 'GEO_GROW_NET_INC',
#        'NET_DEBT_%_CAPITAL', 'WACC_TOTAL_CAPITAL',
#        'TOT_DEBT_TO_TOT_ASSET',
#        'BS_TOT_ASSET', 'RSK_BB_EQTY_PRICE_VOL',
#        'RETURN_ON_INV_CAPITAL', 'CUR_MKT_CAP_val', 'TOT_DEBT_TO_EBITDA_val',
#        'TOTAL_PAYOUT_RATIO_val', 'PE_RATIO_val', 'RETURN_ON_ASSET_val',
#        'RETURN_ON_ASSET_val']

# scaler = StandardScaler()
# scaler.fit(data[cols_to_scale])

# data[cols_to_scale] = scaler.transform(data[cols_to_scale])
# X_2 = data


# In[ ]:


X_train2, X_test2, y_train2, y_test2 = train_test_split(X_2, Y_2, test_size=0.20, random_state=1, stratify=Y)


# ### model III: Lasso

# ### model IV: Balanced Random Forest 

# In[ ]:


model_3_b = BalancedRandomForestClassifier(n_estimators=300)


cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)

scores = cross_val_score(model_3_b, X_2, Y_2, cv=cv, n_jobs=-1)

print(np.mean(scores))


# In[ ]:


model_3_b.fit(X_train2, y_train2)


# In[ ]:


# from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

titles_options = [
    ("Confusion matrix, without normalization", None),
    ("Normalized confusion matrix", "true"),
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        model_3_b,
        X_test2,
        y_test2,
        display_labels=model_3_b.classes_,
        cmap=plt.cm.Blues,
        normalize=normalize,
    )
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()


# In[ ]:


y_true = y_test2
y_pred_2 = model_3_b.predict(X_test2)
target_names = ['-1.', '0.', '1.']

print(classification_report(y_true, y_pred_2, target_names=target_names))


# ## Unified sample

# In[ ]:


financial =numerical_with_rating[['SALES_REV_TURN', 'NET_INCOME', 'GEO_GROW_NET_INC',
       'NET_DEBT_%_CAPITAL', 'WACC_TOTAL_CAPITAL',
       'TOT_DEBT_TO_TOT_ASSET',
       'BS_TOT_ASSET', 'RSK_BB_EQTY_PRICE_VOL',
       'RETURN_ON_INV_CAPITAL', 'CUR_MKT_CAP_val', 'TOT_DEBT_TO_EBITDA_val',
       'TOTAL_PAYOUT_RATIO_val', 'PE_RATIO_val', 'RETURN_ON_ASSET_val',
       'RETURN_ON_ASSET_val', 'CUR_MKT_CAP_is_missing',
       'TOT_DEBT_TO_EBITDA_is_missing', 'TOTAL_PAYOUT_RATIO_is_missing',
       'PE_RATIO_is_missing', 'RETURN_ON_ASSET_is_missing']]
nonfinancial = inputs

frames = [financial, nonfinancial]
mixed_data = pd.concat(frames, axis = 1)
mixed_data


# In[ ]:


y = with_rating_output['Grade'].values
x = mixed_data.values


# In[ ]:


X_train3, X_test3, y_train3, y_test3 = train_test_split(x, y, test_size=0.20, random_state=1, stratify=Y)


# ### model V: Balanced Random Forest

# In[ ]:


model_5 = BalancedRandomForestClassifier(n_estimators=300)


cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)

scores = cross_val_score(model_5, x, y, cv=cv, n_jobs=-1)

print(np.mean(scores))


# In[ ]:


model_5.fit(X_train3,y_train3)


# In[ ]:


# from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

titles_options = [
    ("Confusion matrix, without normalization", None),
    ("Normalized confusion matrix", "true"),
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        model_5,
        X_test3,
        y_test3,
        display_labels=model_3_b.classes_,
        cmap=plt.cm.Blues,
        normalize=normalize,
    )
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()


# In[ ]:


y_true = y_test3
y_pred_2 = model_5.predict(X_test3)
target_names = ['-1.', '0.', '1.']

print(classification_report(y_true, y_pred_2, target_names=target_names))


# ### model VI: Encompassing 

# In[ ]:


from scipy.optimize import minimize
p1 = model_2.predict(X_test)
p2 = model_5.predict(X_test3)


def lossfunction(a):
    d = 0
    for i in range(0,len(y_test3)):
        d_i = (y_test3[i] - (a*p1[i] + (1-a)*p2[i]))**2
        d += d_i
    return d

result = minimize(lossfunction, [0.5], method='L-BFGS-B')
result


# In[ ]:


encomp_predicts =[]

for i in range(0,len(y_test3)): 
               y_hat = 0.52160494*p1[i] + (1-0.52160494)*p2[i]
               encomp_predicts.append(y_hat)


# In[ ]:


tm_encomp_predicts = []

for i in encomp_predicts:
    if i > 0.5:
        i = 1
    elif {i >= 0} & {i<=0.5}:
        i = 0
    else:
        i = -1
    
    tm_encomp_predicts.append(i)
    
pd.DataFrame(tm_encomp_predicts).value_counts()


# In[ ]:


y_true = y_test3
y_pred = tm_encomp_predicts

target_names = ['-1.', '0.', '1.']

print(classification_report(y_true, y_pred, target_names=target_names))


# ## DM tests

# ### for BBDT and BRF

# In[ ]:


from DM_test import dm_test

actual_lst1 = y_test
pred1_lst1 = model_1.predict(X_test)
pred2_lst1 = model_2.predict(X_test)

rt1 = dm_test(actual_lst1,pred1_lst1,pred2_lst1,h = 1, crit="MSE")
print(rt1)


# ### for BRF and Encompassing 

# In[ ]:


actual_lst2 = y_test3
pred1_lst2 = model_5.predict(X_test3)
pred2_lst2 = tm_encomp_predicts

rt2 = dm_test(actual_lst2,pred1_lst2,pred2_lst2,h = 1, crit="MSE")
print(rt2)


# In[ ]:





# In[ ]:




