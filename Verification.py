import pickle 

file = open("One_Class_Models\\1_Model.pickle",'rb')
clf_1 = pickle.load(file)
file = open("One_Class_Models\\2_Model.pickle",'rb')
clf_2 = pickle.load(file)
file = open("One_Class_Models\\3_Model.pickle",'rb')
clf_3 = pickle.load(file)
file = open("One_Class_Models\\4_Model.pickle",'rb')
clf_4 = pickle.load(file)
file = open("One_Class_Models\\5_Model.pickle",'rb')
clf_5 = pickle.load(file)

verify = {}

def main(i, pred):
    l = list()
    l.append(clf_1.predict(i)[0])
    l.append(clf_2.predict(i)[0])
    l.append(clf_3.predict(i)[0])
    l.append(clf_4.predict(i)[0])
    l.append(clf_5.predict(i)[0])

    print(l)