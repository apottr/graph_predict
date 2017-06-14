import pandas as pd
from io import StringIO
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.utils.np_utils import to_categorical
import numpy as np

def create_model(inp_dim,output_labels):
    model = Sequential()
    model.add(Dense(inp_dim,input_shape=(inp_dim,),activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(len(output_labels),activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
    return model


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
training_data = df.loc[:4,'tag1':].values
training_labels = df.loc[:4,'connected_to'].values
testing_data = df.loc[5:,'tag1':].values
testing_labels = df.loc[5:,'connected_to'].values

cat_labels = to_categorical(training_labels)

model = create_model(training_data.shape[1],cat_labels)

model.fit(training_data,training_labels,epochs=500,batch_size=10)
scores = model.evaluate(training_data,training_labels)
print(model.summary())
print("\n{}: {}%".format(model.metrics[0],scores[1]*100))
