# Movie production studio wellness analysis scenario using sentiment

As a hypothetical movie production & distribution company, we want to get an overall picture of viewer sentiment over time for our entire library. This will give the board a high-level view of current tastes in the market. Acting as an early warning system, this could trigger more detailed analysis on whether our productions are aging well enough or if the distribution business is reaching the right audiences.

We are given access to 25,000 historic user reviews of our movies with a binary positive or negative sentiment label, with a further 25,000 labelled reviews for testing.

The board expects a time-series visualisation updated quarterly to be included in preliminary board meeting reports. The board isn't interested in any short-term fluctuations.

## Data
For this scenario we are pretending this dataset represents only movies from our fictional production & distribution studio.
IMDb 50k moview review dataset: `https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data`.

We have a train (25k samples) and test set (25k samples):
- Negatives review has a score <= 4 out of 10,
- Positive reviews have a score >= 7 out of 10. 
- Reviews with a score of 5 or 6 and removed from the data.
- No more than 30 reviews per movie.

An unlabelled set (50k samples), that we will use for our live chart:
- Can be any score range.

## Solution requirement
- Binary classifier to predict sentiment based on incoming reviews.
- A simple time-series chart as a png to be included in a board pack. Business quarter on the x-axis and volume of positive & negative reviews on the y-axis.
- A deployable pipeline that will run batch predictions each quarter, producing the chart.

## Metrics for success 
- 90% Precision: As the outcome of this chart could trigger the board to request a potentially expensive deeper analysis, it is important that we reduce number of inaccurate predictions.
- 98% ROC AUC score: Number of true negatives and positives are equally important to us and we know from the dataset authors that it is balanced equally between positive and negative reviews.

## Detecting drift
When do we know when our training data isn't sufficiently representing our observed data anymore? For this we will set up a semantic drift calculation which sets up this comparison using embeddings.

## Ethics
- Reviews are collected from an english language & american owned movie database so we should expect a western culture bias.
- It's not clear how much research was done to identify 30 reviews per movie as the maximum or what the minimum number could be.
- With little information, there could be significant selection bias on behalf of the researchers when identifying a maximum of 30 reviews per movie. Over what period of time were these collected?
- Is this dataset representative of the level of review polarity on the IMDb website? It's likely selection bias was significant in order to end up with a perfect class balance. Therefore theoretically we shouldn

## Roadmap
1. Data exploration: assess data quality, explore tokens in terms of distibution and types that could impact modelling.
2. Benchmark model: produce a simple classifier for evaluation puposes.
3. Model development: train a more complete classifier to compete with the benchmark.
4. Model evaluation: compare model and benchmark and explore model explainability.
5. Deployment: build a deployable pipeline that will handle batch predictions and output chart.

