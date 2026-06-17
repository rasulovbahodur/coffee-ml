import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

df = pd.read_csv('arabica_data_cleaned.csv')
#print(df.head())
#print(df.shape)
#print(df.columns.tolist())

data = df[['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance', 'Uniformity', 'Clean.Cup', 'Sweetness', 'Cupper.Points', 'Processing.Method']]
data = data[data['Processing.Method'].isin(['Washed / Wet', 'Natural / Dry'])]
data.info()
print(data['Processing.Method'].value_counts())

y = data['Processing.Method']
X = data.drop(columns = ['Processing.Method'])
print(y.shape)
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(X_train.shape)
print(X_test.shape)

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))