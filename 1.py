import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

df = pd.read_csv("offcampus_training.csv")

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2))

features = tfidf.fit_transform(df['text']).toarray()
labels = df.category
print(features.shape)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

X = df.iloc[:,:]
y_train = df['category']

le_name_mapping = {1:0,2:1,3:2,4:3,5:4}

for key in le_name_mapping.keys():
    one_class_dataframe = X.loc[X['category'] == key]
    
    X_train = one_class_dataframe.iloc[:,-1]
    # print(X_train)

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit(X['text'])
    X_train_counts1 = X_train_counts.transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts1)

    clf = IsolationForest().fit(X_train_tfidf, y_train)
    filename = "One_Class_Models\\"+str(key)+"_Model.pickle"
    pickle.dump(clf, open(filename, 'wb'))
    print("Dumped "+str(key)+" Model")

