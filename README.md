# Real-or-Not-NLP-with-Disaster-Tweets

## Project Name: NLP DISASTER TWEETS: EDA, NLP, TENSORFLOW, KERAS

## Problem Statement: 
Welcome to our captivating journey through the realm of sentiment analysis, where we unravel the mysteries of Twitter disaster tweets. In this project, we go beyond the surface, distinguishing between tweets that depict actual disasters and those that metaphorically embody chaos.
##Predict which tweets are about real disasters and which are not
&nbsp; Actual Disaster<br>
&nbsp; Metaphorically Disaster<br>


## Explore the Depths:

### - Introduction
### - Libraries
### - Loading Data
### - Exploratory Data Analysis
   - Analyzing Labels
   - Analyzing Features
     - Sentence Length Analysis
### - Data Cleaning
   - Remove URL
   - Handle Tags
   - Handle Emoji
   - Remove HTML Tags
   - Remove Stopwords and Stemming
   - Remove Useless Characters
   - Wordcloud
### - Final Pre-Processing Data
### - Machine Learning
   - Logistic Regression
   - Naive Bayes
     - Gaussian Naive Bayes
     - Bernoulli Naive Bayes
     - Complement Naive Bayes
     - Multinomial Naive Bayes
   - Support Vector Machine (SVM)
     - RBF kernel SVM
     - Linear Kernel SVM
   - Random Forest
### - Deep Learning
   - Single Layer Perceptron
   - Multi-Layer Perceptron
     - Model 1: Sigmoid + Adam
     - Model 2: Sigmoid + SGD
     - Model 3: Relu + Adam
     - Model 4: Relu + SGD
     - Model 5: Sigmoid + Batch Normalization + Adam
     - Model 6: Sigmoid + Batch Normalization + SGD
     - Model 7: Relu + Dropout + Adam
     - Model 8: Relu + Dropout + SGD

## Prerequisites and Installation

Before embarking on this exciting journey, make sure you have Python and the following libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [Matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [Tensorflow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)

![Requirements](https://user-images.githubusercontent.com/34357926/105755591-87d8af00-5f71-11eb-9bc1-865615ff5759.png)

## Data Overview

Delve into the dataset:

- Size of tweets.csv: 1.53MB
- Number of rows in tweets.csv: 11369

**Features:**
- id: A unique identifier for each tweet
- text: The text of the tweet
- location: The location the tweet was sent from (may be blank)
- keyword: A particular keyword from the tweet (may be blank)
- target: In train.csv only, this denotes whether a tweet is about a real disaster (1) or not (0)

## WordCloud

Witness the visual spectacle of Word Clouds, portraying the frequency of words within the tweets.

![Word Cloud](https://user-images.githubusercontent.com/34357926/105754188-c7060080-5f6f-11eb-9122-71fc6319c040.PNG)

## Results

### Key Performance Index:

Micro f1 score, Macro f1 score, and Micro-Averaged F1-Score (Mean F Score) are the metrics guiding our evaluation. All models are scrutinized based on Accuracy, Precision, Recall, F1-Score, and Time.

![Results](https://user-images.githubusercontent.com/34357926/105753395-a2f5ef80-5f6e-11eb-8d3e-cfda9f9c630b.png)

**Best Performing Models:**
- Support Vector Machine
- Deep Learning (Relu + Adam)
- Deep Learning (Relu + Adam + Dropouts)

## Conclusion

As we conclude our exploration, we discover that Deep Learning Models are susceptible to overfitting and underfitting. Never underestimate the prowess of Machine Learning techniques. Relu and Adam with Dropout emerge as the champions, yet SVM retains its throne as the best in terms of accuracy and training time.

## References:

- [Kaggle - NLP with Disaster Tweets](https://www.kaggle.com/vbmokin/nlp-with-disaster-tweets-cleaning-data)
- [Towards Data Science - NLP for Machine Learning](https://towardsdatascience.com/natural-language-processing-nlp-for-machine-learning-d44498845d5b)
- [Machine Learning Mastery](https://machinelearningmastery.com/)
