import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn .svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


st.title("Machine Learning Without Coding")
st.write("""
Explore different Classifiers
--
#Let's Address the Question 

Which one is the best???               
            
         """)

dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine Dataset"))

classifier_name  = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))


def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y
X, y = get_dataset(dataset_name)
st.write("Shape of Dataset: ", X.shape)
st.write("Number of Classes", len(np.unique(y)))



def add_parameter_ui(classifier_name):
    params = dict()
    if classifier_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif classifier_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("max depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators        
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(classifier_name, params):
    if classifier_name == "KNN":
        clssf =  KNeighborsClassifier(n_neighbors=params["K"])
    elif classifier_name == "SVM":
        clssf = SVC(C=params["C"])
    else:
        clssf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                       max_depth=params["max_depth"], random_state=123)        
    return clssf

clssf = get_classifier(classifier_name, params)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 144)

pca = PCA(2)
X_projected = pca.fit_transform(X_train)

clssf.fit(X_projected, y_train)

X_projected_new = pca.transform(X_test)
b = clssf.predict(X_projected_new)

st.write(f"classifier = {classifier_name}")


fig = plt.figure()
x1 = X_projected_new[:,0]
x2 = X_projected_new[:,1]



plt.scatter(x1, x2, c=b, alpha=0.7, cmap = "winter")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()


st.pyplot(fig)