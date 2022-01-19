from io import StringIO
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

le = LabelEncoder()
scaler = MinMaxScaler()

pd.set_option('display.max_rows', None)

#Please set these as relative paths based on what they are on your local PC
df0=pd.read_csv("ML/FINALPROJECT/TRAINING_SHORTENED.csv", header=None, low_memory=False)
df=pd.read_csv("ML/FINALPROJECT/RandomSample.csv", header=None, low_memory=False)
df_test = pd.read_csv("ML/FINALPROJECT/USB-IDS-1-TEST.csv", header=None, low_memory=False)

df.join(df_test, lsuffix='_caller', rsuffix='_other')

#Data Preprocessing
headers = df0.iloc[0]
df1  = pd.DataFrame(df0.values[1:], columns=headers)
df1.columns = df1.columns.str.replace(' ', '')
columns = df1.columns

def training_dataset_cleanup():
        df.columns = columns
        df.dropna(axis=1, how='all',inplace=True)
        df.drop(['FlowID', 'SourceIP', 'DestinationIP', 'Timestamp', 'FwdAvgBytes/Bulk', 'BwdAvgBytes/Bulk', 
                'FwdAvgBulkRate', 'FwdAvgPackets/Bulk', 'BwdAvgPackets/Bulk', 'BwdAvgBulkRate', 'BwdPSHFlags', 
                'FwdURGFlags', 'BwdURGFlags', 'RSTFlagCount', 'CWEFlagCount', 'ECEFlagCount'], axis=1, inplace=True)

        df[['Attack', 'Defense']] = df['Label'].str.split('-', 1, expand=True)
        df.drop(df[df['Attack'] == '83'].index, inplace = True)

        df.drop('Label', axis=1, inplace=True)
        df.Defense.fillna(value=np.nan, inplace=True)

        df['FlowDuration'] = scaler.fit_transform(df[['FlowDuration']])
        df['TotalLengthofFwdPackets'] = scaler.fit_transform(df[['TotalLengthofFwdPackets']])
        df['TotalLengthofBwdPackets'] = scaler.fit_transform(df[['TotalLengthofBwdPackets']])

        df['Attack'] = le.fit_transform(df['Attack'])
        attack_label_map = dict(zip(le.classes_, le.transform(le.classes_)))
        df['Defense'] = le.fit_transform(df['Defense'])
        defense_label_map = dict(zip(le.classes_, le.transform(le.classes_)))

def test_dataset_cleanup():
        df_test.columns = columns
        df_test.dropna(axis=1, how='all',inplace=True)
        df_test.drop(['FlowID', 'SourceIP', 'DestinationIP', 'Timestamp', 'FwdAvgBytes/Bulk', 'BwdAvgBytes/Bulk', 
                'FwdAvgBulkRate', 'FwdAvgPackets/Bulk', 'BwdAvgPackets/Bulk', 'BwdAvgBulkRate', 'BwdPSHFlags', 
                'FwdURGFlags', 'BwdURGFlags', 'RSTFlagCount', 'CWEFlagCount', 'ECEFlagCount'], axis=1, inplace=True)

        df_test[['Attack', 'Defense']] = df_test['Label'].str.split('-', 1, expand=True)

        df_test.drop(labels=0, axis=0, inplace=True)
        df_test.drop('Label', axis=1, inplace=True)
        df_test.Defense.fillna(value=np.nan, inplace=True)

        df_test['FlowDuration'] = scaler.fit_transform(df_test[['FlowDuration']])
        df_test['TotalLengthofFwdPackets'] = scaler.fit_transform(df_test[['TotalLengthofFwdPackets']])
        df_test['TotalLengthofBwdPackets'] = scaler.fit_transform(df_test[['TotalLengthofBwdPackets']])

        df_test['Attack'] = le.fit_transform(df_test['Attack'])
        attack_label_map = dict(zip(le.classes_, le.transform(le.classes_)))
        df_test['Defense'] = le.fit_transform(df_test['Defense'])
        defense_label_map = dict(zip(le.classes_, le.transform(le.classes_)))

training_dataset_cleanup()

bins = ['BENIGN', 'Hulk', 'Slowhttptest' ,'Slowloris', 'TCPFlood']
classes = [0, 1, 2, 3, 4]

info_gain_columns = ['Init_Win_bytes_forward', 'FwdHeaderLength', 'FwdIATMean', 'TotalFwdPackets', 'SubflowFwdPackets',
                     'BwdPackets/s', 'PacketLengthMean', 'FlowPackets/s', 'AveragePacketSize', 'FlowIATMax',
                     'SourcePort', 'PacketLengthVariance', 'FwdIATMax', 'FwdPackets/s', 'PacketLengthStd',
                     'FlowIATMean', 'FlowIATStd', 'FlowBytes/s' ,'SubflowBwdBytes', 'FwdIATStd', 'TotalLengthofBwdPackets',
                     'Init_Win_bytes_backward', 'AvgBwdSegmentSize', 'BwdPacketLengthMean', 'BwdPacketLengthMax',
                     'BwdPacketLengthStd', 'FwdPacketLengthMax', 'BwdHeaderLength' ,'SubflowBwdPackets', 'MaxPacketLength',
                     'AvgFwdSegmentSize', 'FwdPacketLengthStd', 'FwdPacketLengthMean', 'TotalBackwardPackets', 'DestinationPort',
                     'FwdIATMin', 'SubflowFwdBytes', 'FlowDuration']

igc = ['Init_Win_bytes_forward', 'FwdHeaderLength', 'FwdIATMean', 'TotalFwdPackets', 'SubflowFwdPackets',
                     'BwdPackets/s', 'PacketLengthMean', 'FlowPackets/s', 'AveragePacketSize', 'FlowIATMax',
                     'SourcePort', 'PacketLengthVariance', 'FwdIATMax', 'FwdPackets/s', 'PacketLengthStd',
                     'FlowIATMean', 'FlowIATStd', 'FlowBytes/s' ,'SubflowBwdBytes', 'FwdIATStd']

X = df[igc]
y = df['Attack']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Information Gain

mutual_info = mutual_info_classif(X_train, y_train)
# print(mutual_info)

mutual_info = pd.Series(mutual_info)
mutual_info.index = X_train.columns
print(mutual_info.sort_values(ascending=False))

#Decision Tree Clasifier

depth = 15

clf = DecisionTreeClassifier(criterion="entropy", max_depth=depth)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Max depth:", depth)
print("Accuracy:",  metrics.accuracy_score(y_test, y_pred))

# Logistic Regression

clf = LogisticRegression()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",  metrics.accuracy_score(y_test, y_pred))

# Confusion Matrix Visualization

print(metrics.confusion_matrix(y_test, y_pred, labels=classes))
metrics.plot_confusion_matrix(clf, X_test, y_test)
plt.show()

# K-Nearest Neighbour

# creating list of K for KNN
k_list = list(range(1,50,2))
# creating list of cv scores
cv_scores = []

# perform 10-fold cross validation
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

MSE = [1 - x for x in cv_scores]

plt.figure()
plt.figure(figsize=(15,10))
plt.title('The optimal number of neighbors', fontsize=20, fontweight='bold')
plt.xlabel('Number of Neighbors K', fontsize=15)
plt.ylabel('Misclassification Error', fontsize=15)
sn.set_style("whitegrid")
plt.plot(k_list, MSE)

plt.show()

best_k = k_list[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d." % best_k)


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("KNN Accuracy: ", metrics.accuracy_score(y_test, y_pred))

#Decision Tree Visualization
dot_data = StringIO()

export_graphviz(clf, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names = info_gain_columns, class_names=bins)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('data.png')
Image(graph.create_png())