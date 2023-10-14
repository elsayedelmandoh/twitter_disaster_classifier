# Disaster Tweet Classification

## Table of Contents

- [Project Overview](#project-overview)
- [Data](#data)
- [Data Exploration](#data-exploration)
- [Data Preparation](#data-preparation)
- [Model Building](#model-building)
- [Model Evaluation](#model-evaluation)
- [Save Model](#save-model)
- [How to Use](#how-to-use)
- [Author](#author)

## Project Overview

Natural disasters can have a significant impact on people's lives, and social media has become an important source of information during such events. This project aims to build a machine learning model that can classify tweets as disaster-related or not, based on their text content. The project utilizes Natural Language Processing (NLP) techniques to preprocess and analyze the text and trains several machine learning models to predict the classification of each tweet.


## Data

The dataset used for this project is sourced from [Kaggle](https://www.kaggle.com/competitions/nlp-getting-started/data). It consists of labeled tweets with information about whether they are related to a disaster or not.

## Data Exploration

The project starts with importing and exploring the dataset, including visualizations of the distribution of the number of tokens and the average token length in each text.

## Data Preparation

Text data preprocessing involves handling missing data, noise removal (removing URLs and stopwords), and balancing the target classes.

## Model Building

The project utilizes a pipeline that includes TF-IDF vectorization and a Naive Bayes classifier. Hyperparameter tuning is performed using Grid Search to find the best combination of hyperparameters.

## Model Evaluation

The final model is evaluated on a test set, and metrics such as accuracy, a classification report, and a confusion matrix are provided. The results are visualized for better interpretation.

## Save Model

The best-performing model is saved for future use. It can be loaded and used to classify new tweets.

## How to Use

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/disaster-tweet-classification.git

2. **Run the notebook:**
   ```bash
   python twitter_disaster_classifier.ipynb


## Author
  Elsayed Elmandoh : [Linkedin](https://www.linkedin.com/in/elsayed-elmandoh-77544428a/).

