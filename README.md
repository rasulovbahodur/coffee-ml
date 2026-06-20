# Coffee Processing Classifier

Predicting how a coffee was processed — **washed** or **natural** — from its sensory cupping scores and country of origin.

This is my first machine learning project. I built it to understand how a real ML workflow fits together, and I learned a lot doing it. I used Claude to help me understand the concepts and work through every step — it took me from completely lost to a working model I understand, and it worked.

## The question

Coffee tasters often say they can tell whether a coffee was washed or natural just by tasting it. I wanted to see if that holds up: can a model recover the processing method from a coffee's measured attributes?

## The data

The Coffee Quality Institute (CQI) database — around 1,300 professionally graded arabica coffees. I narrowed it to the two main processing methods, leaving **1,063 coffees** (812 washed, 251 natural).

- **Features:** 10 sensory cupping scores (aroma, flavor, acidity, body…) plus country of origin.
- **Target:** processing method (washed vs natural).

## Approach

- Trained on 80% of the data, tested on a held-out 20% (stratified to preserve the class balance).
- Random Forest with balanced class weights to handle the imbalance (washed is ~76% of the data).
- One-hot encoded the country of origin so the model could use it alongside the numeric scores.

## Results — and what I learned digging in

**The imbalance trap.** My first model hit 77% accuracy but was a fake — it just guessed "washed" almost every time, catching only 2 of 50 naturals. Raw accuracy looked fine only because washed coffees dominate. Measuring **balanced accuracy** (each class weighted equally) exposed it: a real ~52%, barely above a coin flip.

**Breaking the ceiling.** On taste scores alone, the model plateaued around **64% balanced accuracy** — washed and natural overlap a lot in flavor. Adding **country of origin** jumped it to **~85% raw / ~77% balanced.** Better features beat a fancier model.

**What drives it.** Feature importance showed the single biggest predictor is whether the coffee is from **Brazil** (the dominant natural-process country), followed by the taste scores, each contributing a smaller, roughly equal amount. So the model reads processing partly through geography and partly through flavor — and geography is the stronger signal.

| Features | Raw accuracy | Balanced accuracy |
|---|---|---|
| Taste scores only | 74% | 63% |
| Taste scores + country | 85% | 77% |

## Try it

Interactive Streamlit app: enter a coffee's ten cupping scores, pick its country, and it predicts washed or natural with a confidence estimate.

Live demo: https://coffee-ml-2nwqfr8pc4xdhbb3ojy3cq.streamlit.app/

## How to run

```
pip install pandas scikit-learn streamlit
python main.py          # trains the model and prints the evaluation
streamlit run app.py    # launches the interactive app
```

## What I learned

How to load and clean real data, why train/test splits and stratification matter, how class imbalance can fool you, why a single accuracy number can't be trusted, how to one-hot encode categorical features, how to read a confusion matrix and feature importances, and how to deploy a model as an app. Using Claude to explain the *why* behind each step — instead of just handing me code — is what made it stick.