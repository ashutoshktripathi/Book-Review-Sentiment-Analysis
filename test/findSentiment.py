import pandas as pd
from lte250classifer import classify as torchClassify
from gte250classifer import classify as tfClassify

def loadData(path):
    data = pd.read_csv(path)
    data["len"] = data.apply(lambda x : len(str(x["review"]).split()) , axis = 1)
    return data

df = loadData('test_data.csv')
df = df.reset_index()

# check for any nans since it will brak the model
# remove now and late will merge back

dfnan = df[df.review.isna()]

# will mark all the sentiment as neutral since has no test or empty text
dfnan["sentiment"] = "neutral"

df = df[df.review.isna() == False]


dflte250 = df[df.len <= 250]
dfgte250 = df[df.len > 250]
print('shape of data', df.shape)


if len(dfgte250) > 0:
        print('shape of gte250', dfgte250.shape)
        rev = list(dfgte250["review"])
        summ = list(dfgte250["summary"])
        sentiments = tfClassify(rev,summ,True)
        dfgte250["sentiment"] = sentiments


if len(dflte250) > 0:
        print('shape of lte250', dflte250.shape)
        rev = list(dflte250["review"])
        sentiments = torchClassify(rev)
        dflte250["sentiment"] = sentiments


total = pd.concat([dfgte250, dflte250, dfnan], ignore_index=True)
total = total.sort_values('index')
total.to_csv('results.csv', index=False)