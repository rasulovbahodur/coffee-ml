# Coffee Processing Classifier

Predicting how a coffee was processed — **washed** or **natural** — from its sensory cupping scores.

This is my first machine learning project. I built it to understand how a real ML workflow fits together, and I learned a lot doing it. I used Claude to help me understand the concepts and work through every step — it took me from completely lost to a working model I actually understand, and it worked.

## The question

Coffee tasters often say they can tell whether a coffee was washed or natural just by tasting it. I wanted to see if that holds up: can a model recover the processing method from flavor scores alone?

## The data

The Coffee Quality Institute (CQI) database — around 1,300 professionally graded arabica coffees. I narrowed it to the two main processing methods, leaving **1,063 coffees** (812 washed, 251 natural).

- **Features:** 10 sensory cupping scores — aroma, flavor, aftertaste, acidity, body, balance, uniformity, clean cup, sweetness, cupper points.
- **Target:** processing method (washed vs natural).

## Approach

- Trained on 80% of the data, tested on a held-out 20% (stratified so both splits keep the same washed/natural ratio).
- Used a Random Forest classifier with balanced class weights to handle the imbalance (washed is ~76% of the data).

## Results — and the part I found most interesting

My first version hit **77% accuracy** — which looked good until I checked the confusion matrix. It was just guessing "washed" almost every time, catching only **2 of 50** natural coffees. The accuracy looked fine only because washed coffees dominate the dataset.

After adding balanced class weights, the accuracy *dropped* — but the model started telling the two apart. The honest way to measure that is **balanced accuracy** (how well it does on each class, averaged so the rare class counts equally):

| Model | Raw accuracy | Balanced accuracy |
| Logistic Regression (naive) | 77% | 52% |
| Logistic Regression (balanced) | 63% | 64% |
| Random Forest (balanced) | 74% | 63% |

Takeaways:
- **Raw accuracy can lie** on imbalanced data — the 77% model was a coin flip in disguise (52% balanced).
- A more powerful model (Random Forest) didn't beat the simpler one on balanced accuracy. The ceiling (~64%) comes from the data, not the algorithm.
- **Conclusion:** taste profile predicts processing method meaningfully but not strongly — clear signal above a coin flip, but washed and natural overlap enough in flavor that it's far from solved.

(The final `main.py` uses the Random Forest; the others were part of the comparison.)

## How to run

```
pip install pandas scikit-learn
python main.py
```

It prints the test accuracy, the confusion matrix, and a sample prediction. You can plug your own coffee's 10 cupping scores into the bottom of `main.py` to get a predicted method.

## What I learned

How to load and clean real data, why train/test splits and stratification matter, how class imbalance can fool you, why a single accuracy number can't be trusted, and how to read a confusion matrix. Using Claude to explain the *why* behind each step — instead of just handing me code — is what made it stick.