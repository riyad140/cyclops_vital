# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 14:24:44 2024

@author: imrul
"""

import pickle
import matplotlib.pyplot as plt

# Function to read a pickle file
def read_pickle_file(file_path):
    """
    Reads a pickle file and returns the content.

    Parameters:
        file_path (str): Path to the pickle file.
    
    Returns:
        data: The data loaded from the pickle file.
    """
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            print("Pickle file read successfully!")
            return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except pickle.UnpicklingError:
        print("Error: The file could not be unpickled. It may not be a valid pickle file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

#%%
# Example usage
if __name__ == "__main__":
    file_path = r"W:\raspberrypi\photos\Beta\B004\2024-12-02\S009\wbc-results-20241203_153228-3part_fullcrop_test\S009-df_features.pickle"  # Replace with the path to your pickle file
    # file_path = r"W:\raspberrypi\photos\Beta\B006\2024-12-02\S033_PREC_RUN3\wbc-results-20241202_133000-3part_fullcrop\S033_PREC_RUN3-df_features.pickle"
    df = read_pickle_file(file_path)
    
    if df is not None:
        print("Data loaded from pickle file:")
        filtered_df = df[df['prediction'] != -1]
        
        print(filtered_df)
        
    plt.figure()
    plt.scatter(filtered_df['mean_intensity_darkfield'], filtered_df['mean_intensity_red'],s = 1, c = filtered_df['prediction'], )
    plt.xlim([0,1000])
    plt.ylim([0,1000])
    plt.xlabel('mean_intensity_darkfield')
    plt.ylabel('mean_intensity_red')
        
#%%
    array_2d = filtered_df[['mean_intensity_darkfield', 'mean_intensity_red']].to_numpy()
    pred = filtered_df['prediction'].to_numpy()
    
    

#%%
# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load the dataset (Iris dataset as an example)
# iris = load_iris()
# X = iris.data  # Features
# y = iris.target  # Labels

X,y = array_2d, pred

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Create and train the K-NN model
k = 41  # Number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Step 4: Make predictions on the test set
y_pred = knn.predict(X_test)

# Step 5: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the K-NN model with k={k}: {accuracy:.2f}")

# Step 6: Predict for new data (optional)
# sample_data = [[5.9, 3.0, 5.1, 1.8]]  # Example input
# prediction = knn.predict(sample_data)
# print(f"Predicted class for {sample_data}: {iris.target_names[prediction][0]}")




#%%



# import numpy as np
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.preprocessing import StandardScaler

# # Step 1: Load the dataset
# # Replace with your dataset
# # data = load_iris()
# # X, y = data.data, data.target

# X,y = array_2d, pred

# # Step 2: Preprocess the data (scaling)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Step 3: Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# # Step 4: Train the SVM model
# # Choose a kernel: 'linear', 'poly', 'rbf', 'sigmoid'
# svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
# svm_model.fit(X_train, y_train)

# # Step 5: Make predictions
# y_pred = svm_model.predict(X_test)

# # Step 6: Evaluate the model
# print("Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred))
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# # Optional: Visualize decision boundaries (2D datasets only)
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap

# def plot_decision_boundaries(X, y, model):
#     cmap = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
#     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    
#     Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
    
#     plt.contourf(xx, yy, Z, alpha=0.8, cmap=cmap)
#     plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='viridis')
#     plt.title("Decision Boundary of SVM")
#     plt.show()

# # Visualize on first two features (if applicable)
# plot_decision_boundaries(X_test[:, :2], y_test, svm_model)




#%% validation with different dataset

# file_path = r"W:\raspberrypi\photos\Beta\B006\2024-12-02\S033_PREC_RUN3\wbc-results-20241202_133000-3part_fullcrop\S033_PREC_RUN3-df_features.pickle"
file_path = r"W:\raspberrypi\photos\Beta\B004\2024-12-02\S009\wbc-results-20241203_153228-3part_fullcrop_test\S009-df_features.pickle" # Replace with the path to your pickle file
    # file_path = r"W:\raspberrypi\photos\Beta\B004\2024-12-02\S009\wbc-results-20241203_153228-3part_fullcrop_test\S009-df_features.pickle"  # Replace with the path to your pickle file
df = read_pickle_file(file_path)

if df is not None:
    print("Data loaded from pickle file:")
    filtered_df = df[df['prediction'] != -1]
    
    print(filtered_df)
    
plt.figure()
plt.scatter(filtered_df['mean_intensity_darkfield'], filtered_df['mean_intensity_red'],s = 1, c = filtered_df['prediction'], )
plt.xlim([0,1000])
plt.ylim([0,1000])
plt.xlabel('mean_intensity_darkfield')
plt.ylabel('mean_intensity_red')

array_2d = filtered_df[['mean_intensity_darkfield', 'mean_intensity_red']].to_numpy()
pred = filtered_df['prediction'].to_numpy()


X_test,y_test = array_2d, pred

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the K-NN model with k={k}: {accuracy:.2f}")



plt.figure()
plt.scatter(X_test[:,0],X_test[:,1],s=1,c=y_pred)
plt.xlim([0,1000])
plt.ylim([0,1000])
plt.xlabel('mean_intensity_darkfield')
plt.ylabel('mean_intensity_red')
plt.title('KNN predicted labels')

# Step 6: Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

import pandas as pd

# Example dataset
labels = pd.Series(pred)

# Calculate percentages
label_percentages = labels.value_counts(normalize=True) * 100

# Display the results
print('Before')
print(label_percentages)

labels = pd.Series(y_pred)

# Calculate percentages
label_percentages = labels.value_counts(normalize=True) * 100

print('After')
print(label_percentages)
    