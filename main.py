import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix

data=pd.read_excel('rssi_data.xlsx')
print(data)
data['COOrdinates'] = data['COOrdinates'].apply(lambda x: tuple(float(i) for i in x[1:-1].split(',')))

data[['X', 'Y']] = pd.DataFrame(data['COOrdinates'].tolist(), columns=['X', 'Y'])

X=data[['RSSI', 'X', 'Y']]
Y=data['Room']

# print(X)

X_train, X_test, y_train, Y_test= train_test_split(X, Y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

# print(classification_report(Y_test, y_pred))
# print(confusion_matrix(Y_test, y_pred))

new_rssi = -15
new_coordinate = (24.525, 26.633)

new_data = pd.DataFrame([[new_rssi, new_coordinate[0], new_coordinate[1]]], columns=['RSSI', 'X', 'Y'])

predicted_room = knn.predict(new_data)[0]

print(f'The predicted room is: {predicted_room}')

# plt.figure(figsize=(10, 6))

# Scatter plot of the training data points
# colors = {'hall': 'orange', 'room1': 'blue', 'room2': 'green'}
# for room in y_train.unique():
#     plt.scatter(X_train[y_train == room]['X'], X_train[y_train == room]['Y'], color=colors[room], label=f'{room}', alpha=0.6)
#
# # Scatter plot of the new data point
# plt.scatter(new_data['X'], new_data['Y'], color='yellow', edgecolor='black', s=100, label='New Point')
#
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.title('KNN Room Prediction Scatter Plot')
# plt.legend()
# plt.show()
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
#
# knn=KNeighborsClassifier(n_neighbors=3)
# knn.fit(X_train, y_train)
#
# new_data = [[-34, (24.624, 26.6180)]]
# new_data = scaler.transform(new_data)
# prediction = knn.predict(new_data)
# print(prediction)