import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('arabica_data_cleaned.csv')

data = df[['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance', 'Uniformity', 'Clean.Cup', 'Sweetness', 'Cupper.Points', 'Country.of.Origin', 'Processing.Method']]
data = data[data['Processing.Method'].isin(['Washed / Wet', 'Natural / Dry'])]
data = data.dropna(subset=['Country.of.Origin'])

y = data['Processing.Method']
X = data.drop(columns = ['Processing.Method'])
X = pd.get_dummies(X, columns=['Country.of.Origin'])

model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X, y)

st.title("Coffee Processing Classifier")
st.write("Enter a coffee's ten sensory cupping scores (each from 0 to 10), then click Predict. The model guesses whether it was washed or natural from its flavor profile.")

features = ['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance', 'Uniformity', 'Clean.Cup', 'Sweetness', 'Cupper.Points']

st.caption("Scores use the 0–10 SCA cupping scale — the standard professional coffee-tasting scale.")

scores = []
for f in features:
    scores.append(st.slider(f, 0.0, 10.0, 7.5, step=0.25))

country = st.selectbox("Country of origin", sorted(df['Country.of.Origin'].dropna().unique()))

if st.button("Predict"):
    new_coffee = pd.DataFrame([scores + [country]], columns=features + ['Country.of.Origin'])
    new_coffee = pd.get_dummies(new_coffee, columns=['Country.of.Origin'])
    new_coffee = new_coffee.reindex(columns=X.columns, fill_value=0)
    result = model.predict(new_coffee)[0]
    confidence = model.predict_proba(new_coffee).max()
    st.success(f"Prediction: {result} — {confidence:.0%} confident")

st.caption("This model is right about two-thirds of the time — washed and natural coffees overlap a lot in flavor, so treat it as an educated guess.")