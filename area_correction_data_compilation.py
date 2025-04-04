# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 12:24:08 2025

@author: imrul
"""

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def fit_regression_and_evaluate(x, y):
    """
    Fit a regression line to the input arrays and calculate R^2 and MAE.

    Parameters:
        x (array-like): Independent variable values.
        y (array-like): Dependent variable values.

    Returns:
        dict: A dictionary with slope, intercept, R^2, and MAE values.
    """
    # Ensure x and y are numpy arrays
    x = np.array(x).reshape(-1, 1)  # Reshape x for sklearn
    y = np.array(y)

    # Fit the regression model
    model = LinearRegression()
    model.fit(x, y)

    # Make predictions
    y_pred = model.predict(x)

    # Calculate metrics
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    # Plot the data and regression line
    plt.figure()
    plt.scatter(x, y, color='blue', label='Data points')
    plt.plot(x, y_pred, color='red', label='Regression line')
    plt.xlabel('Independent Variable (x)')
    plt.ylabel('Dependent Variable (y)')
    plt.title('Regression Line with Data')
    text_str = f"R^2 = {r2:.2f}\nMAE = {mae:.2f}"
    plt.gca().text(0.5, 0.95, text_str, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    plt.legend()
    plt.ylim([150,400])
    plt.show()

    # Return results
    return {
        "slope": model.coef_[0],
        "intercept": model.intercept_,
        "r2": r2,
        "mae": mae
    }
#%%

df=pd.DataFrame()



#AS3

# plt_sysmex = [361.67,361.67,361.67,251.67,251.67,251.67,241.7,241.7,241.7,193,193,193,247,247,247]
# plt_ht = [370.39,316.92,315.50,208.92,227.27,227.27,217.19,201.93,201.86,208.9,217.4,186,211.04,247.74,232.30]
# plt_ht_revised = [390.5,333.5,326.5,212.6,226.4,232.8,223,212.9,211.6,220.5,214,216.7,211.1,242.5,227.9]


#B006

# plt_sysmex = [215.00,215.00,215.00,372.00,372.00,372.00,278.00,278.00,278.00, 207.00,207.00,207.00]
# plt_ht = [232.22,236.1,235.39,342.33,335.97,355.74,280.21,275.57,297.86,216.69,216.69,213.52]
# plt_ht_revised = [220.8,229.7,231.7,310.4,307.2,330.3,255.7,244.4,270.8,207.8,212.9,208]


# AS01

# plt_sysmex = [196.00,196.00,196.00,204.00,204.00,290.00,290.00,290.00,236.00,236.00,236.00]

# plt_ht = [214.82,217.39,169.04,201.11,225.21,359.27,306.33,312.73,257.91,215.63,238.31]

# plt_ht_revised = [206.7,215.5,166,203.8,222.2,327.4,326.4,347.3,243,213.9,232.7]


# AS01 MC 


# plt_sysmex = [261,230,282,208,241,358,169,297,251,268,236]
# plt_ht = [233.98,226.22,245.63,176.45,190.92,260.45,140.61,256.92,196.34,225.51,174.08]
# plt_ht_revised = [    231.5,223.4,238.6,181.5,212.5,295.6,159.2,241.7,226.5,243.7,221.5]



# SuperSet
# for area
# plt_sysmex = [361.67,361.67,361.67,251.67,251.67,251.67,241.7,241.7,241.7,193,193,193,247,247,247,215.00,215.00,215.00,372.00,372.00,372.00,278.00,278.00,278.00, 207.00,207.00,207.00,
#               196.00,196.00,196.00,204.00,204.00,290.00,290.00,290.00,236.00,236.00,236.00,261,230,282,208,241,358,169,297,251,268,236]
# plt_ht = [370.39,316.92,315.50,208.92,227.27,227.27,217.19,201.93,201.86,208.9,217.4,186,211.04,247.74,232.30,232.22,236.1,235.39,342.33,335.97,355.74,280.21,275.57,297.86,216.69,216.69,213.52,
#           214.82,217.39,169.04,201.11,225.21,359.27,306.33,312.73,257.91,215.63,238.31,233.98,226.22,245.63,176.45,190.92,260.45,140.61,256.92,196.34,225.51,174.08]
# plt_ht_revised=[390.5,333.5,326.5,212.6,226.4,232.8,223,212.9,211.6,220.5,214,216.7,211.1,242.5,227.9,220.8,229.7,231.7,310.4,307.2,330.3,255.7,244.4,270.8,207.8,212.9,208,
#                 206.7,215.5,166,203.8,222.2,327.4,326.4,347.3,243,213.9,232.7,231.5,223.4,238.6,181.5,212.5,295.6,159.2,241.7,226.5,243.7,221.5] 


# for dedup
plt_sysmex = [361.67,361.67,361.67,251.67,251.67,251.67,241.7,241.7,241.7,193,193,193,247,247,247,215,215,215,372,372,372,278,278,278,207,207,207]
plt_ht = [370.39,316.92,315.50,208.92,227.27,227.27,217.19,201.93,201.86,208.9,217.4,186,211.04,247.74,232.30,232.2,236.1,235.4,342.3,336,355.7,280.2,275.6,297.9,216.7,216.7,213.5]
plt_ht_revised=[389.57,354.46,348.2,248.1,239.3,229.3,230.2,249.96,259.12,226,242.8,201.1,243,274.1,266.01,258.6,262.4,268.5,373.9,382.1,391.2,314.3,294.4,312.9,240.7,252.9,250.8] 



# plt.figure()
# plt.plot(plt_sysmex,plt_ht,'o')


stats1 = fit_regression_and_evaluate(plt_sysmex,plt_ht)
stats2 = fit_regression_and_evaluate(plt_sysmex,plt_ht_revised)


#%%
dict_plt_sysmex = {'AS3':[361.67,361.67,361.67,251.67,251.67,251.67,241.7,241.7,241.7,193,193,193,247,247,247],
                    'B06':[215.00,215.00,215.00,372.00,372.00,372.00,278.00,278.00,278.00, 207.00,207.00,207.00],
                    'AS1': [196.00,196.00,196.00,204.00,204.00,290.00,290.00,290.00,236.00,236.00,236.00],
                    'AS1_MC': [261,230,282,208,241,358,169,297,251,268,236]   
                    }


dict_plt_ht = {'AS3':[370.39,316.92,315.50,208.92,227.27,227.27,217.19,201.93,201.86,208.9,217.4,186,211.04,247.74,232.30],
                'B06':[232.22,236.1,235.39,342.33,335.97,355.74,280.21,275.57,297.86,216.69,216.69,213.52],
                'AS1':[214.82,217.39,169.04,201.11,225.21,359.27,306.33,312.73,257.91,215.63,238.31],
                'AS1_MC':[233.98,226.22,245.63,176.45,190.92,260.45,140.61,256.92,196.34,225.51,174.08]    
    }

dict_plt_ht_revised = {'AS3':[390.5,333.5,326.5,212.6,226.4,232.8,223,212.9,211.6,220.5,214,216.7,211.1,242.5,227.9],
                        'B06':[220.8,229.7,231.7,310.4,307.2,330.3,255.7,244.4,270.8,207.8,212.9,208],
                        'AS1':[206.7,215.5,166,203.8,222.2,327.4,326.4,347.3,243,213.9,232.7],
                        'AS1_MC':[231.5,223.4,238.6,181.5,212.5,295.6,159.2,241.7,226.5,243.7,221.5]
       
    }


stats = stats2

r2 = stats['r2']
mae = stats['mae']

plt.figure()
plt.plot(plt_sysmex,stats['slope']*np.array(plt_sysmex)+stats1['intercept'],label = 'fit')
plt.plot(np.sort(plt_sysmex),np.sort(plt_sysmex),'k--',label='y=x line')
for key in dict_plt_sysmex.keys():
    plt.plot(dict_plt_sysmex[key],dict_plt_ht_revised[key],'o',label = key)
plt.legend()
text_str = f"R^2 = {r2:.2f}\nMAE = {mae:.2f}"
plt.gca().text(0.5, 0.95, text_str, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
plt.ylim([140,400])
plt.xlabel('PLT COUNT SYSMEX')
plt.ylabel('PLT COUNT HT')


#%%

dict_cv = {
    'AS3': [9.36,4.79,4.27,7.96,8.00],
    'AS3_': [6.12,3.94,6,7.93,6.18],
    
    'B06': [0.88,2.93,4.13,0.85],
    'B06_': [1.9,2.26,3.62,2.63],
    
    # 'AS1': [13.57,7.99,8.86,8.92],
    # 'AS1_':[13.47,6.11,3.53,6.42],
    
    
    }

item1 = dict_cv['B06']
item2 = dict_cv['B06_']

plt.figure()
plt.bar(np.arange(len(item1)),item1,alpha=0.7, label = 'default')
plt.bar(np.arange(len(item2)),item2,alpha=0.7, label = 'dual offset')
plt.legend()
plt.ylim([0,11])
plt.ylabel('CV %')