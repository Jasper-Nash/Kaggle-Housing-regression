# Predicting House Prices in Ames, Iowa

**Team:** Jasper Nash, Jaden Kwon, Akhil Lewis

Our entry for the [Regression for house prices](https://www.kaggle.com/competitions/regression-for-house-prices)
Kaggle competition: predict the sale price of 260 houses in Ames, Iowa. Submissions are scored on RMSE in dollars.

## Final model

A scikit-learn `Pipeline` that fills in missing values with `SimpleImputer`, encodes quality ratings with
`OrdinalEncoder` and `Neighborhood`/`SaleCondition` with `OneHotEncoder`, then fits a `LinearRegression` to
**log(SalePrice)** using 16 features.

| Model | Test RMSE (25% held out) |
|---|---|
| Predict the mean price | $79,207 |
| `LinearRegression` on `OverallQual` only | $47,945 |
| All 16 features | $26,421 |
| All features + polynomial degree 2 | $22,527 |
| All features + polynomial degree 3 (overfits) | $38,914 |
| **All features, log(price) (final)** | **$20,783** |

| Kaggle submission | Score |
|---|---|
| First submission (13-feature Ridge regression) | $25,571 |
| Final model (`submissions/submission.csv`) | _[add score]_ |

## Our process

1. **Explore:** price is driven mostly by overall quality and size. One huge, unfinished house sold far below
   its size and quality, so we dropped houses over 4,500 sq ft from training.
2. **Build up from baselines** with a train/test split, adding one idea at a time.
3. **Model complexity:** polynomial features help at degree 2, then overfit badly at degree 3.
4. **Predict log(price):** with a log target, a plain linear model beats the polynomial models, so we keep the
   simpler one.

The notebook walks through each step.

## Repository contents

| Path | What it is |
|---|---|
| `house_prices.ipynb` | The full analysis, executed with outputs |
| `submissions/submission.csv` | Our final Kaggle predictions |
| `requirements.txt` | Python packages needed |

## Running it

1. Download `train.csv` and `test.csv` from the competition's
   [Data tab](https://www.kaggle.com/competitions/regression-for-house-prices/data) (folder `ames/`) into `data/`.
   The data isn't included in this repo because it's subject to the competition rules.
2. `pip install -r requirements.txt`
3. Open `house_prices.ipynb` and run all cells.
