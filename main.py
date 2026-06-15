import pandas as pd
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
print(X.shape)
print(y.shape)