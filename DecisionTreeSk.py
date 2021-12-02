import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split #import train_test_split function
from sklearn import metrics
from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus
from six import StringIO


col_names = ['pregnant', 'glucose', 'bp', 'skin','insulin','bmi','pedigree','age','label']
pima = pd.read_csv("diabetes.csv", header=None, names = col_names)

print(pima.head())

#split datasets into features and target variable
feature_cols = ['pregnant', 'glucose', 'bp', 'skin' ,'insulin','bmi','pedigree','age']
X = pima[feature_cols] #Features
print(X)
y = pima.label #Target variables

#spliting dataset into training set and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=1)
#70% training and 30% testing

#Creating Decision Tree classifier object
clf = DecisionTreeClassifier(criterion='entropy', max_depth = 4)

#Train Decision tree Classifier
clf = clf.fit(X_train,y_train)

#predict the response for test dataset
y_pred = clf.predict(X_test)


#Model Accuracy
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

dot_data = StringIO()

export_graphviz(clf, out_file=dot_data,filled=True, rounded=True,special_characters=True,feature_names=feature_cols,class_names=['0','1'])

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('diabetes.png')
Image(graph.create_png())
