# House Prices Project

Team: Jasper Nash, Jaden Kwon, Akhil Lewis

This is our project for the [Regression for house prices](https://www.kaggle.com/competitions/regression-for-house-prices)
Kaggle competition. We predicted the sale price of houses in Ames, Iowa. It's scored with RMSE in dollars.

## What we did

- looked at the data and took out one weird house (really big but sold cheap because it wasn't finished)
- started with simple baselines (guessing the average, then just OverallQual)
- used 16 features in a sklearn pipeline (SimpleImputer, OrdinalEncoder, OneHotEncoder, LinearRegression)
- tried polynomial degrees 1-3. degree 2 helped, degree 3 overfit
- predicted log(price) instead of price, which worked best

Final model: linear regression on log(price) with 16 features.

| Model | Test RMSE |
|---|---|
| guess the average | 79,207 |
| just OverallQual | 47,945 |
| all 16 features | 26,421 |
| polynomial degree 2 | 22,527 |
| polynomial degree 3 | 38,914 |
| log(price) (final) | 20,783 |

| Kaggle submission | Score |
|---|---|
| First submission (Ridge regression) | 25,571 |
| Final model | _[add score]_ |

## Files

- `house_prices.ipynb` - the notebook with all our code
- `submissions/submission.csv` - our final predictions

To run it, put `train.csv` and `test.csv` from the Kaggle data tab into a `data/` folder and run the notebook.
