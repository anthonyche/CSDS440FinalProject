import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split #import train_test_split function
from sklearn import metrics
from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus
from six import StringIO
from sklearn import datasets

data = pd.read_csv('volcanoes.data',header = None, sep=',', names=[i for i in range(228)])
#data1 = np.array(data)
feature_col = [i for i in range(1,227)]
label_col = [i for i in range(227,228)]
X = data[feature_col]
print(X)
y = data[label_col]
print(y)


# data = datasets.load_iris()
# data = datasets.load_diabetes()
# X = data.data
# y = data.target
#spliting dataset into training set and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=1)
#70% training and 30% testing


#Creating Decision Tree classifier object
clf = DecisionTreeClassifier(criterion='entropy', max_depth = 12)

#Train Decision tree Classifier
clf = clf.fit(X_train,y_train)

#predict the response for test dataset
y_pred = clf.predict(X_test)


#Model Accuracy
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

dot_data = StringIO()

export_graphviz(clf, out_file=dot_data,filled=True, rounded=True,special_characters=True,class_names=['0','1'])

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('volcanoes.png')
Image(graph.create_png())




