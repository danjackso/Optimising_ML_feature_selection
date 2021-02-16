#Shell working directory:
#Documents/4th_Year/Learning_Python/Regression/Linear_regression/

import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import sys
import pyswarms as pso
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit

# Importing selected modules

os.system('clear')
print('{} Running'.format(__file__))
print(datetime.now())
warnings.filterwarnings('ignore')

# Importing data and separating it into X and Y numpy arrays.
data=pd.read_csv('Wine.csv')
data_X, data_y = data.iloc[:,:-1].values, data.iloc[:,-1].values


#One hot encoding the Y-value. In this case, a one-hot encoding can be applied
# to the integer representation. This is where the integer encoded variable is
# removed and a new binary variable is added for each unique integer value.
encode=OneHotEncoder(sparse=False)
data_y=encode.fit_transform(data_y.reshape(-1, 1))


def estimator_accuracy_test(x, X_data, Y_data, return_std=False):
    # This function has inputs of a binary array of seledcted features. For
    # example, [0,1,1] would be for the second and third features to be used and
    # [0,1,0] would be for just the second feature to be used. The length of the
    # x array must equal the column length of the X_data input where X_data is
    # the array of the raw features. # Y_data is the one-hot-encoded y data for
    # the ML algorthim to be trained to. This will return the accuracy score of
    # the ML algorthim with the selected features. The accuracy score is
    # Calculated by k-fold shuffling the training set and the test set 'n' number
    # of times. The returned fucntion accuracy score is the average of the
    # k-fold accuracy scores.

    #Code for the optimser.
    n_particles = x.shape[0]
    j, standard_deviation, local_accuracy = [],[],[]

    #K_fold shuffling
    rs = ShuffleSplit(n_splits=15, test_size=.3, random_state=0)
    for i in range(n_particles):
        idx = list(np.where(x[i] == 1)[0])

        local_accuracy=[]
        for train_index, test_index in rs.split(X_data):

            # Indexing the Raw x_data features using matrix column indexing. Also,
            # selecting the k-folded selected training features.

            X_train=X_data[train_index[:,None],idx]
            X_test=X_data[test_index[:,None],idx]

            Y_train=Y_data[train_index,:]
            Y_test=Y_data[test_index,:]


            # Standard scaling the X_data. Because the X_test data is meant to be
            # unknown before prediction, the scaling transformation is based on
            # the X_train data.

            scaler = StandardScaler()
            X_train=scaler.fit_transform(X_train)
            X_test=scaler.transform(X_test)

            #Fitting and calculation of the accuracy_score
            clf = DecisionTreeClassifier(random_state=0)
            clf = clf.fit(X_train, Y_train)

            accuracy=accuracy_score(Y_test,clf.predict(X_test))
            local_accuracy.append(accuracy)

        standard_deviation.append(np.std(local_accuracy))
        j.append(np.mean(local_accuracy))

    if return_std==True:
        return np.array(j), np.array(standard_deviation)
    else:
        return -np.array(j)

def Feature_Optimise(X_train,Y_train):
    # Optimising feature selection using particle swarm global optimisation
    # Algorthim.
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 2, 'p': 2}
    optimizer = pso.discrete.binary.BinaryPSO(n_particles=5,
                                              dimensions=X_train.shape[1],
                                              options=options,
                                              ftol=-1e-03)

    # Inputing Keywords and setup of the optimser.
    kwargs={'X_data':X_train,'Y_data':Y_train,'return_std':False}
    cost, pos = optimizer.optimize(estimator_accuracy_test, iters=100, n_processes=4, **kwargs)

    idx = list(np.where(pos > 0)[0])

    return idx, np.expand_dims(pos, axis=0)

features, pos = Feature_Optimise(data_X, data_y)

# Plotting the Results of the feature selection.
all_features_mean, all_features_std=estimator_accuracy_test(np.ones((1,data_X.shape[1])),data_X,data_y,return_std=True)
optimisation_mean, optimisation_std =estimator_accuracy_test(pos,data_X,data_y, return_std=True)

x=['All Features', 'Optimised Features']
x_pos = np.arange(len(x))
y=[all_features_mean[0], optimisation_mean[0]]
error = [all_features_std[0], optimisation_std[0]]

fig, ax = plt.subplots()
ax.bar(x_pos, y, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
for datapoint in zip(x_pos,y):
    ax.text(datapoint[0],0.01,np.round(datapoint[1],3),ha='center')

ax.set_xticks(x_pos)
ax.set_xticklabels(x)
ax.yaxis.grid(True)
ax.set_ylim([0,1])
ax.set_title('Accuracy of ML Algorithm with Optimised Feature Selection')
plt.tight_layout()
plt.show()
