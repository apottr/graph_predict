import pandas as pd
from io import StringIO
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.utils.np_utils import to_categorical

def create_model(output_labels):
    model = Sequential()
    model.add(Dense(3,input_shape=(3,),activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=output_labels,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adadelta')
    return model

def create_one_hot(labels):
    largest = labels[0]
    out = []
    for i in range(1,len(labels)):
        if labels[i] > largest:
            largest = i

    for i in labels:
        #if(i == 0):
            #out.append([]
        out.append([(0 if index != i else 1) for index in range(0,largest+1)])
    return out

csv = """
id,connected_to,tag1,tag2,tag3
0,4,1,0,0
1,2,0,1,0
2,0,1,1,0
3,4,0,0,1
4,0,1,0,1
5,1,0,1,1
6,2,1,1,1
"""

df = pd.read_csv(StringIO(csv),index_col='id')
training_data = df.loc[:4,'tag1':]
training_labels = df.loc[:4,'connected_to']
testing_data = df.loc[5:,'tag1':]
testing_labels = df.loc[5:,'connected_to']

print(to_categorical([int(i) for i in training_labels]))
model = create_model(to_categorical([int(i) for i in training_labels]))

print(training_data)
print(training_labels)
print(testing_data)
print(testing_labels)
