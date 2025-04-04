# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 14:24:44 2024

@author: imrul
"""

import pickle
import matplotlib.pyplot as plt
import pandas as pd


def z_score_normalization(data, columns=None):
    """
    Apply Z-score normalization to the specified columns of the dataset.
    
    Parameters:
    - data: pandas DataFrame or numpy array
    - columns: List of column names to normalize (if using a DataFrame)
    
    Returns:
    - normalized_data: pandas DataFrame or numpy array with normalized values
    """
    if isinstance(data, pd.DataFrame):
        if columns is None:
            columns = data.columns
        normalized_data = data.copy()
        for column in columns:
            mean = data[column].mean()
            std = data[column].std()
            normalized_data[column] = (data[column] - mean) / std
    elif isinstance(data, np.ndarray):
        # Assuming the data is 2D (rows: samples, columns: features)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        normalized_data = (data - mean) / std
    else:
        raise ValueError("Input data must be a pandas DataFrame or a numpy array.")
    
    return normalized_data


def min_max_scaling(data, columns=None, feature_range=(0, 1)):
    """
    Apply Min-Max scaling to the specified columns of the dataset.

    Parameters:
    - data: pandas DataFrame or numpy array
    - columns: List of column names to scale (if using a DataFrame)
    - feature_range: Tuple (min, max) for the target scaling range
    
    Returns:
    - scaled_data: pandas DataFrame or numpy array with scaled values
    """
    min_val, max_val = feature_range
    
    if isinstance(data, pd.DataFrame):
        if columns is None:
            columns = data.columns
        scaled_data = data.copy()
        for column in columns:
            col_min = data[column].min()
            col_max = data[column].max()
            scaled_data[column] = (data[column] - col_min) / (col_max - col_min) * (max_val - min_val) + min_val
    elif isinstance(data, np.ndarray):
        # Assuming the data is 2D (rows: samples, columns: features)
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        scaled_data = (data - data_min) / (data_max - data_min) * (max_val - min_val) + min_val
    else:
        raise ValueError("Input data must be a pandas DataFrame or a numpy array.")
    
    return scaled_data

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


        
def build_training_set(filePaths, normalization = False):
    training_df = [] #pd.DataFrame()
    
    for file_path in filePaths:
        df = pd.read_csv(file_path)
        # if df is not None:
        #     print("Data loaded from pickle file:")
        #     filtered_df = df[df['prediction'] != -1]
        df1=df.copy()
        if normalization is True:
            df1 = min_max_scaling(df, columns = ['area', 'meanintensitygreen', 'meanintensityred','intensitymediandarkfield'])
            print('Normalizing data')
    
        # return normalized_data
        training_df.append(df1)
    
    dataset_df = pd.concat(training_df, ignore_index= True)
    
    
    # if normalization is True:
    #     normalized_data = min_max_scaling(dataset_df, columns = ['area', 'meanintensitygreen', 'meanintensityred','intensitymediandarkfield'])
    #     print('Normalizing data')
    #     return normalized_data
    
    # else:
    return dataset_df


def build_test_set(filePaths, normalization = False):
    training_df = [] #pd.DataFrame()
    
    for file_path in filePaths:
        df = read_pickle_file(file_path)
        if df is not None:
            print("Data loaded from pickle file:")
            df = df[df['prediction'] != -1]
            df1=df.copy()
        if normalization is True:
            df1 = min_max_scaling(df, columns = ['mean_intensity_green','mean_intensity_darkfield', 'mean_intensity_red','area'])
            print('Normalizing data')
    
        # return normalized_data
        training_df.append(df1)
    
    dataset_df = pd.concat(training_df, ignore_index= True)
    
    
    # if normalization is True:
    #     normalized_data = min_max_scaling(dataset_df, columns = ['area', 'meanintensitygreen', 'meanintensityred','intensitymediandarkfield'])
    #     print('Normalizing data')
    #     return normalized_data
    
    # else:
    return dataset_df


#%%

file_paths = [
    
                r"W:\raspberrypi\photos\Offsite\2024-11-27\s035-C\wbc-results-20241127_153241-3part_fullcrop\Manual_gating\LabeledPopulation.csv",
                r"W:\raspberrypi\photos\Offsite\2024-11-27\s004\wbc-results-20241127_155556-3part_fullcrop\Manual_gating\LabeledPopulation.csv",
                r"W:\raspberrypi\photos\Offsite\2024-11-26\s041\wbc-results-20241127_102244-3part_fullcrop\Manual_gating\LabeledPopulation.csv", # high DR
                r"W:\raspberrypi\photos\Offsite\2024-11-26\s032\wbc-results-20241127_101402-3part_fullcrop\Manual_gating\LabeledPopulation.csv"
              
              ]

normalization_flag = True
              
              
filtered_df = build_training_set(file_paths, normalization=normalization_flag)      
label_mapping = {'Lymph': 1, 'Mono': 2, 'Neutro': 0, 'Eos': 3}       

filtered_df['Prediction'] = filtered_df['Label'].map(label_mapping)

colors = {0: 'blue', 1: 'darkred', 2: 'green', 3: 'orange'}
point_colors = [colors[label] for label in filtered_df['Prediction'] ]
              
              # Replace with the path to your pickle file
    # file_path = r"W:\raspberrypi\photos\Beta\B006\2024-12-02\S033_PREC_RUN3\wbc-results-20241202_133000-3part_fullcrop\S033_PREC_RUN3-df_features.pickle
            
    
plt.figure()
plt.scatter(filtered_df['intensitymediandarkfield'], filtered_df['meanintensityred'],s = 1, c = point_colors )
# plt.xlim([0,1000])
# plt.ylim([0,1000])
plt.xlabel('mean_intensity_darkfield')
plt.ylabel('mean_intensity_red')
plt.title('KNN TRAINING DATASET')

plt.figure()
plt.scatter(filtered_df['area'], filtered_df['meanintensityred'],s = 1, c = point_colors )
# plt.xlim([0,1000])
# plt.ylim([0,1000])
plt.xlabel('area')
plt.ylabel('mean_intensity_red')
plt.title('KNN TRAINING DATASET')


plt.figure()
plt.scatter(filtered_df['intensitymediandarkfield'], filtered_df['meanintensityred']/filtered_df[ 'meanintensitygreen'],s = 1, c = point_colors )
# plt.xlim([0,1000])
# plt.ylim([0,1000])
plt.xlabel('mean_intensity_darkfield')
plt.ylabel('mean_intensity_red/green')
plt.title('KNN TRAINING DATASET')

plt.figure()
plt.scatter(filtered_df['area'], filtered_df['meanintensityred']/filtered_df[ 'meanintensitygreen'],s = 1, c = point_colors )
# plt.xlim([0,1000])
# plt.ylim([0,1000])
plt.xlabel('area')
plt.ylabel('mean_intensity_red/green')
plt.title('KNN TRAINING DATASET')
            
    
array_2d = filtered_df[['intensitymediandarkfield', 'meanintensityred','area']].to_numpy()
pred = filtered_df['Prediction'].to_numpy()

#%%

# colors = {
#     1: 'blue',
#     2 : 'green',
#     3: 'yellow',
#     0 : 'red'
# }

# plt.figure()
# plt.hist(filtered_df['area'],bins = 255)


plt.figure()
for label, color in colors.items():
    subset = filtered_df[filtered_df['Prediction'] == label]  # Filter rows for the specific label
    plt.hist(subset['area'], bins=255, color=color, label=subset['Label'].values[0], alpha=0.6)
plt.title('Area Histogram')
plt.legend()

plt.figure()
for label, color in colors.items():
    subset = filtered_df[filtered_df['Prediction'] == label]  # Filter rows for the specific label
    plt.hist(subset['meanintensityred'], bins=255, color=color, label=subset['Label'].values[0], alpha=0.6)
plt.title('RFL Histogram')
plt.legend()
plt.figure()
for label, color in colors.items():
    subset = filtered_df[filtered_df['Prediction'] == label]  # Filter rows for the specific label
    plt.hist(subset['meanintensityred']/subset['meanintensitygreen'], bins=255, color=color,label=subset['Label'].values[0], alpha=0.6)
plt.title('RFL/GFL Histogram')
plt.legend()

plt.figure()
for label, color in colors.items():
    subset = filtered_df[filtered_df['Prediction'] == label]  # Filter rows for the specific label
    plt.hist(subset['intensitymediandarkfield'], bins=255, color=color, label=subset['Label'].values[0], alpha=0.6)
plt.title('DF Histogram')
plt.legend()


#%%
# # Example usage
# if __name__ == "__main__":
#     file_path = r"W:\raspberrypi\photos\Beta\B004\2024-12-02\S009\wbc-results-20241203_153228-3part_fullcrop_test\S009-df_features.pickle"  # Replace with the path to your pickle file
#     # file_path = r"W:\raspberrypi\photos\Beta\B006\2024-12-02\S033_PREC_RUN3\wbc-results-20241202_133000-3part_fullcrop\S033_PREC_RUN3-df_features.pickle"
#     df = read_pickle_file(file_path)
    
#     if df is not None:
#         print("Data loaded from pickle file:")
#         filtered_df = df[df['prediction'] != -1]
        
#         print(filtered_df)
        
#     plt.figure()
#     plt.scatter(filtered_df['mean_intensity_darkfield'], filtered_df['mean_intensity_red'],s = 1, c = filtered_df['prediction'], )
#     plt.xlim([0,1000])
#     plt.ylim([0,1000])
#     plt.xlabel('mean_intensity_darkfield')
#     plt.ylabel('mean_intensity_red')
        
#%%

    
    

#%%
# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Step 1: Load the dataset (Iris dataset as an example)
# iris = load_iris()
# X = iris.data  # Features
# y = iris.target  # Labels

X,y = array_2d, pred

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Step 3: Create and train the K-NN model
k = 3  # Number of neighbors

# k_arr = np.arange(1,501,20)
k_arr = [21]

for k in k_arr:
    
    print(f'---------------------------{k}------------------------------')

    knn = KNeighborsClassifier(n_neighbors=k,metric='manhattan', weights= 'uniform')
    knn.fit(X_train, y_train)
    
    # Step 4: Make predictions on the test set
    y_pred = knn.predict(X_test)
    
    # Step 5: Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of the K-NN model with k={k}: {accuracy:.2f}")
    
    
    file_paths = [
            r"W:\raspberrypi\photos\Beta\B006\2024-11-18\S04_PREC_RUN3\wbc-results-20241119_103641-3part_fullcrop\S04_PREC_RUN3-df_features.pickle"
            ]
    
    
    filtered_df = build_test_set(file_paths, normalization = normalization_flag)
    
    # Replace with the path to your pickle file
    # file_path = r"W:\raspberrypi\photos\Beta\B004\2024-12-02\S009\wbc-results-20241203_153228-3part_fullcrop_test\S009-df_features.pickle"  # Replace with the path to your pickle file
    # df = read_pickle_file(file_path)
    
    # if df is not None:
    #     print("Data loaded from pickle file:")
    #     filtered_df = df[df['prediction'] != -1]
        
    #     print(filtered_df)
        
    
        
    array_2d = filtered_df[['mean_intensity_darkfield', 'mean_intensity_red','area']].to_numpy()
    pred = filtered_df['prediction'].to_numpy()
    
    
    X_test,y_test = array_2d, pred
    
    y_pred = knn.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"TEST Accuracy of the K-NN model with k={k}: {accuracy:.2f}")
    
    labels = pd.Series(y_test)

    # Calculate percentages
    label_percentages = labels.value_counts(normalize=True) * 100
    
    print('Before')
    print(label_percentages)
    
    
    
    labels = pd.Series(y_pred)

    # Calculate percentages
    label_percentages = labels.value_counts(normalize=True) * 100
    
    print('After')
    print(label_percentages)
    
    plt.figure()
    plt.scatter(X_test[:,0],X_test[:,1],s=1,c=y_test)
    # plt.xlim([0,1000])
    # plt.ylim([0,1000])
    plt.xlabel('mean_intensity_darkfield')
    plt.ylabel('mean_intensity_red')
    plt.title('Original')
    
    
    plt.figure()
    plt.scatter(X_test[:,0],X_test[:,1],s=1,c=y_pred)
    # plt.xlim([0,1000])
    # plt.ylim([0,1000])
    plt.xlabel('mean_intensity_darkfield')
    plt.ylabel('mean_intensity_red')
    plt.title('After KNN predicted labels')
    
    plt.figure()
    plt.scatter(X_test[:,2],X_test[:,1],s=1,c=y_pred)
    # plt.xlim([0,1000])
    # plt.ylim([0,1000])
    plt.xlabel('area')
    plt.ylabel('mean_intensity_red')
    plt.title('After KNN predicted labels')
        
#%% Normalizing Test Dataset

# file_paths = [
#         r"W:\raspberrypi\photos\Beta\B006\2024-11-18\S04_PREC_RUN1\cycImages\wbc-results-20241118_125043-3part_fullcrop\cycImages-df_features.pickle",
#         r"W:\raspberrypi\photos\Beta\B006\2024-11-18\S04_PREC_RUN2\wbc-results-20241118_133908-3part_fullcrop\S04_PREC_RUN2-df_features.pickle",
#         r"W:\raspberrypi\photos\Beta\B006\2024-11-18\S04_PREC_RUN3\wbc-results-20241119_103641-3part_fullcrop\S04_PREC_RUN3-df_features.pickle",
        
        
#         r"W:\raspberrypi\photos\Beta\B006\2024-12-02\S033_PREC_RUN1_REPEAT\wbc-results-20241202_164643-3part_fullcrop\S033_PREC_RUN1_REPEAT-df_features.pickle",
#         r"W:\raspberrypi\photos\Beta\B006\2024-12-02\S033_PREC_RUN2\wbc-results-20241202_132353-3part_fullcrop\S033_PREC_RUN2-df_features.pickle",
#         r"W:\raspberrypi\photos\Beta\B006\2024-12-02\S033_PREC_RUN3\wbc-results-20241202_133000-3part_fullcrop\S033_PREC_RUN3-df_features.pickle"
#         ]

# file_paths = [
# r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS1\PBSvsPBSD_October24\2024-10-07\S1222-1_PBS_IM5.1_AS1\wbc-results-20241009_093635-3part_fullcrop\S1222-1_PBS_IM5.1_AS1-df_features.pickle",
# r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS1\PBSvsPBSD_October24\2024-10-07\S1223_MC_PBS_IM5.1_AS1\wbc-results-20241007_151146-3part_fullcrop\S1223_MC_PBS_IM5.1_AS1-df_features.pickle",
# r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS1\PBSvsPBSD_October24\2024-10-08\S1224-1_PBS_IM5.1_AS1\wbc-results-20241008_144723-3part_fullcrop\S1224-1_PBS_IM5.1_AS1-df_features.pickle",
# r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS1\PBSvsPBSD_October24\2024-10-09\MC_S1226_PBS_IM5.1_AS1\wbc-results-20241010_131016-3part_fullcrop\MC_S1226_PBS_IM5.1_AS1-df_features.pickle",
# r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS1\PBSvsPBSD_October24\2024-10-09\S1225-1_PBS_IM5.1_AS1\wbc-results-20241009_124348-3part_fullcrop\S1225-1_PBS_IM5.1_AS1-df_features.pickle",
# r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS1\PBSvsPBSD_October24\2024-10-10\S1229_PBS_IM5.1_AS1\wbc-results-20241010_164940-3part_fullcrop\S1229_PBS_IM5.1_AS1-df_features.pickle",
# r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS1\PBSvsPBSD_October24\2024-10-10\S1230-1_PBS_IM5.1_AS1\wbc-results-20241010_170338-3part_fullcrop\S1230-1_PBS_IM5.1_AS1-df_features.pickle",
# r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS1\PBSvsPBSD_October24\2024-10-11\MC_S1231_PBS_IM5.1_AS1_repeat\wbc-results-20241011_130819-3part_fullcrop\MC_S1231_PBS_IM5.1_AS1_repeat-df_features.pickle",
    
#     ]

# file_paths = [
# r"W:\raspberrypi\photos\Beta\B006\2024-11-18\S04_PREC_RUN1\cycImages\wbc-results-20241118_125043-3part_fullcrop\cycImages-df_features.pickle",
# r"W:\raspberrypi\photos\Beta\B006\2024-11-18\S04_PREC_RUN2\wbc-results-20241118_133908-3part_fullcrop\S04_PREC_RUN2-df_features.pickle",
# r"W:\raspberrypi\photos\Beta\B006\2024-11-18\S04_PREC_RUN3\wbc-results-20241119_103641-3part_fullcrop\S04_PREC_RUN3-df_features.pickle",
#  r"W:\\raspberrypi\\photos\\Beta\\B006\\2024-12-02\\S033_PREC_RUN1_REPEAT\\wbc-results-20241202_164643-3part_fullcrop\\S033_PREC_RUN1_REPEAT-df_features.pickle",
#     r"W:\\raspberrypi\\photos\\Beta\\B006\\2024-12-02\\S033_PREC_RUN2\\wbc-results-20241202_132353-3part_fullcrop\\S033_PREC_RUN2-df_features.pickle",
#     r"W:\\raspberrypi\\photos\\Beta\\B006\\2024-12-02\\S033_PREC_RUN3\\wbc-results-20241202_133000-3part_fullcrop\\S033_PREC_RUN3-df_features.pickle"
#     ]


file_paths = [
    r"W:\raspberrypi\photos\Beta\B004\2024-12-17\s024_0\wbc-results-20241217_142211-3part_fullcrop\s030_0-df_features.pickle"
    
    ]


filtered_df = build_test_set(file_paths, normalization = False)


array_2d = filtered_df[['mean_intensity_darkfield', 'mean_intensity_red','area']].to_numpy()
pred = filtered_df['prediction'].to_numpy()


X_test,y_test = array_2d, pred
    
plt.figure()
plt.scatter(X_test[:,0],X_test[:,1],s=1,c=y_test)
# plt.xlim([0,1000])
# plt.ylim([0,1000])
plt.xlabel('mean_intensity_darkfield')
plt.ylabel('mean_intensity_red')
plt.title('Original-1')    
    
plt.figure()
plt.scatter(X_test[:,2],X_test[:,1],s=1,c=y_test)
# plt.xlim([0,1000])
# plt.ylim([0,1000])
plt.xlabel('area')
plt.ylabel('mean_intensity_red')
plt.title('Original-2')   


# Step 6: Predict for new data (optional)
# sample_data = [[5.9, 3.0, 5.1, 1.8]]  # Example input
# prediction = knn.predict(sample_data)
# print(f"Predicted class for {sample_data}: {iris.target_names[prediction][0]}")

plt.figure()
plt.hist(filtered_df['mean_intensity_red'],bins = 255)

#%%

plt.figure()
for label, color in colors.items():
    subset = filtered_df[filtered_df['prediction'] == label]  # Filter rows for the specific label
    plt.hist(subset['area'], bins=255, color=color, label=subset['prediction'].values[0], alpha=0.6)
plt.title('Area Histogram')
plt.legend()

plt.figure()
for label, color in colors.items():
    subset = filtered_df[filtered_df['prediction'] == label]  # Filter rows for the specific label
    plt.hist(subset['mean_intensity_red'], bins=255, color=color, label=subset['prediction'].values[0], alpha=0.6)
plt.title('RFL Histogram')
plt.legend()
plt.figure()
for label, color in colors.items():
    subset = filtered_df[filtered_df['prediction'] == label]  # Filter rows for the specific label
    plt.hist(subset['mean_intensity_red']/subset['mean_intensity_green'], bins=255, color=color,label=subset['prediction'].values[0], alpha=0.6)
plt.title('RFL/GFL Histogram')
plt.legend()

plt.figure()
for label, color in colors.items():
    subset = filtered_df[filtered_df['prediction'] == label]  # Filter rows for the specific label
    plt.hist(subset['mean_intensity_darkfield'], bins=255, color=color, label=subset['prediction'].values[0], alpha=0.6)
plt.title('DF Histogram')
plt.legend()




#%%
plt.figure()
plt.scatter(filtered_df['area'],filtered_df['mean_intensity_red']/filtered_df['mean_intensity_green'],s=1,c=y_test)
# plt.xlim([0,1000])
# plt.ylim([0,1000])
plt.xlabel('area')
plt.ylabel('mean_intensity_red/green')
plt.title('area vs R/G')   

plt.figure()
plt.scatter(filtered_df['mean_intensity_darkfield'],filtered_df['mean_intensity_red']/filtered_df['mean_intensity_green'],s=1,c=y_test)
# plt.xlim([0,1000])
# plt.ylim([0,1000])
plt.xlabel('mean_intensity_darkfield')
plt.ylabel('mean_intensity_red/green')
plt.title('area vs R/G')   

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

# # file_path = r"W:\raspberrypi\photos\Beta\B006\2024-12-02\S033_PREC_RUN3\wbc-results-20241202_133000-3part_fullcrop\S033_PREC_RUN3-df_features.pickle"
# file_path = r"W:\raspberrypi\photos\Offsite\2024-11-27\s035-A\wbc-results-20241127_152050-3part_fullcrop\s035-df_features.pickle"# Replace with the path to your pickle file
#     # file_path = r"W:\raspberrypi\photos\Beta\B004\2024-12-02\S009\wbc-results-20241203_153228-3part_fullcrop_test\S009-df_features.pickle"  # Replace with the path to your pickle file
# df = read_pickle_file(file_path)

# if df is not None:
#     print("Data loaded from pickle file:")
#     filtered_df = df[df['prediction'] != -1]
    
#     print(filtered_df)
    
# plt.figure()
# plt.scatter(filtered_df['mean_intensity_darkfield'], filtered_df['mean_intensity_red'],s = 1, c = filtered_df['prediction'], )
# plt.xlim([0,1000])
# plt.ylim([0,1000])
# plt.xlabel('mean_intensity_darkfield')
# plt.ylabel('mean_intensity_red')

# array_2d = filtered_df[['mean_intensity_darkfield', 'mean_intensity_red']].to_numpy()
# pred = filtered_df['prediction'].to_numpy()


# X_test,y_test = array_2d, pred

# y_pred = knn.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy of the K-NN model with k={k}: {accuracy:.2f}")



# plt.figure()
# plt.scatter(X_test[:,0],X_test[:,1],s=1,c=y_pred)
# plt.xlim([0,1000])
# plt.ylim([0,1000])
# plt.xlabel('mean_intensity_darkfield')
# plt.ylabel('mean_intensity_red')
# plt.title('KNN predicted labels')

# # Step 6: Evaluate the model
# print("Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred))
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# # import pandas as pd

# # Example dataset
# labels = pd.Series(pred)

# # Calculate percentages
# label_percentages = labels.value_counts(normalize=True) * 100

# # Display the results
# print('Before')
# print(label_percentages)

# labels = pd.Series(y_pred)

# # Calculate percentages
# label_percentages = labels.value_counts(normalize=True) * 100

# print('After')
# print(label_percentages)
    