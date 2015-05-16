#!/usr/bin/python

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('./input/train.csv')
test  = pd.read_csv('./input/test.csv')

y = train[['label']]
x = train.drop('label', axis=1)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(x,np.ravel(y))
pred = clf.predict(test)

submission = pd.DataFrame({"ImageId":test.index+1, "Label":pred})
submission.to_csv('rf_python_digits.csv', index=False)
