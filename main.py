import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from Verification import main


df = pd.read_csv("offcampus_training.csv")
df1 = pd.read_csv("offcampus_test.csv")
# d = list()

# for i, row in df.iterrows():
#     print(i)
#     l = list(row['text'].split(' '))
#     d.append(l)

# df1 = pd.DataFrame({'id': df['id'], 'category': df['category'], 'text': d})
# print(df1.head())


import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(8,6))
# df.groupby('category').id.count().plot.bar(ylim=0)
# plt.show()

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2))

features = tfidf.fit_transform(df['text']).toarray()
labels = df.category
print(features.shape)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPClassifier

# X_train, X_test, y_train, y_test = train_test_split(df['text'], df['category'], random_state = 0)
X_train = df['text']
y_train = df['category']
X_test = df1['text']


# X_train, X_test, y_train, y_test = train_test_split(df['Consumer_Complaint'], df['Product'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit(X_train)
X_train_counts1 = X_train_counts.transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts1)

X_test_counts = X_train_counts.transform(X_test)
X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)


# MLP = MLPClassifier().fit(X_train_tfidf, y_train)
LSVC = LinearSVC().fit(X_train_tfidf, y_train)
# print(X_test_tfidf)
for i in X_test_tfidf:
    y_pred = LSVC.predict(i)
    main(i, y_pred)
    exit()
    

exit()
results = pd.DataFrame({'id':df1['id'], 'category':y_pred})
results.to_csv("SVCRBF.csv", header=True)