import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Load the full CQI coffee dataset
df = pd.read_csv('arabica_data_cleaned.csv')
#print(df.head())
#print(df.shape)
#print(df.columns.tolist())

# Keep the 10 sensory scores (features) + processing method (target),
# then filter down to just the two main methods
data = df[['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance', 'Uniformity', 'Clean.Cup', 'Sweetness', 'Cupper.Points', 'Processing.Method']]
data = data[data['Processing.Method'].isin(['Washed / Wet', 'Natural / Dry'])]
#data.info()
#print(data['Processing.Method'].value_counts())

# Split into target (y = the answer) and features (X = the taste scores)
y = data['Processing.Method']
X = data.drop(columns = ['Processing.Method'])
#print(y.shape)
#print(X.shape)

# Hold back 20% for testing; stratify keeps the washed/natural ratio in both splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#print(X_train.shape)
#print(X_test.shape)

# Train a Random Forest; class_weight='balanced' stops it from just guessing the majority class
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))   # overall accuracy on the held-out test set

# Confusion matrix: how it did per class (rows = actual, columns = predicted)
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))

# Predict the processing method for a brand-new coffee from its 10 cupping scores
new_coffee = pd.DataFrame(
    [[7.8, 7.5, 7.4, 7.6, 7.5, 7.5, 10, 10, 9, 7.5]],   # the 10 scores, in this order
    columns=['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body',
             'Balance', 'Uniformity', 'Clean.Cup', 'Sweetness', 'Cupper.Points']
    )
print(model.predict(new_coffee))