#!/usr/bin/env python3
"""
Natural Podcast Generator for DACS Study Guide
Creates conversational narration like NotebookLM with natural pauses and flow
"""

import asyncio
import edge_tts
import os
import re
import subprocess
from pathlib import Path

# Natural conversational voices
HOST_A = "en-US-AndrewNeural"   # Male - Warm, Confident, Authentic
HOST_B = "en-US-AvaNeural"      # Female - Expressive, Caring, Friendly

# Audio output directory
AUDIO_DIR = Path(__file__).parent / "static" / "audio"

# ============================================
# EXPANDED DETAILED PODCAST SCRIPTS
# ============================================

PODCAST_SCRIPTS = {
    # ========== DEFINITIONS SECTION ==========
    'definitions-ml-basics': {
        'title': 'ML Basics - Complete Guide',
        'section': 'definitions',
        'dialogue': [
            ('A', "Welcome back to the DACS Study Guide podcast! Today we're doing a deep dive into Machine Learning fundamentals. And I really want to emphasize how important these concepts are."),
            ('B', "Absolutely. These are the building blocks for everything else in this course. If you understand these terms deeply, the rest just falls into place naturally."),
            ('A', "So let's start with the big question. What exactly is Machine Learning? And I mean really understand it, not just memorize a definition."),
            ('B', "Machine Learning is a subset of Artificial Intelligence. But here's what makes it special and different from traditional programming."),
            ('A', "In traditional programming, you write explicit rules. If this condition, then do that. If email contains lottery, mark as spam. Very specific, very manual."),
            ('B', "But what happens when spammers get clever? They write lott ery with a space. Or use zero instead of O. Your rules break."),
            ('A', "Exactly! This is where Machine Learning shines. Instead of writing rules, you show the computer examples. Lots and lots of examples."),
            ('B', "You give it thousands of spam emails and thousands of legitimate emails. And you say... learn the patterns. Figure out what makes spam, spam."),
            ('A', "The computer then builds a mathematical model. A function that can look at a new email it's never seen and predict whether it's spam or not."),
            ('B', "And this is so powerful because it can find patterns humans might miss. Subtle combinations of words, timing patterns, sender behaviors."),
            ('A', "Now, there are different types of Machine Learning. The first major category is Supervised Learning. And the name is really descriptive."),
            ('B', "Think of it like learning with a teacher who has all the answers. You have labeled data. That means for every input, you know the correct output."),
            ('A', "So if you're predicting house prices, you have historical data. You know what features each house had, AND what it actually sold for."),
            ('B', "The sale price is your label. Your ground truth. Your target variable. The algorithm learns the relationship between features and this target."),
            ('A', "Supervised Learning splits into two main types. Classification and Regression. This distinction is really important."),
            ('B', "Classification is when you're predicting a category. A discrete label. Spam or not spam. Cat or dog. Fraud or legitimate. Disease or healthy."),
            ('A', "The output is one of a set of predefined categories. There's no in-between. An email is either spam or it isn't."),
            ('B', "Regression is different. You're predicting a continuous number. House price, temperature, stock value, someone's age."),
            ('A', "The output can be any number in a range. A house could sell for three hundred thousand, or three hundred one thousand, or any value."),
            ('B', "Now let's talk about Unsupervised Learning. This is fundamentally different because there are no labels."),
            ('A', "No teacher with answer keys. You just have data, and you're asking the algorithm to find hidden structure or patterns."),
            ('B', "Customer segmentation is the classic example. You have data about your customers but no predefined groups."),
            ('A', "The algorithm clusters similar customers together. Maybe it finds high-value frequent buyers, occasional browsers, bargain hunters."),
            ('B', "You didn't tell it what groups to find. It discovered them from the patterns in the data. That's unsupervised learning."),
            ('A', "Anomaly detection is another huge application. Find things that don't fit the normal pattern. Fraud, network intrusions, manufacturing defects."),
            ('B', "Okay, let's define some more fundamental terms. Features. This is absolutely core vocabulary."),
            ('A', "A feature is an input variable. A piece of information you use to make predictions. In a spreadsheet, think of each column as a feature."),
            ('B', "If you're predicting loan defaults, your features might include income, credit score, employment length, debt to income ratio, number of previous loans."),
            ('A', "Each of these gives the model information to work with. More relevant features often lead to better predictions."),
            ('B', "You'll also hear features called predictors, attributes, independent variables, or just inputs. Same concept, different names."),
            ('A', "Now the target. Also called the label, the dependent variable, the output, or ground truth. This is what you're trying to predict."),
            ('B', "In our loan example, the target might be a binary yes or no. Did they default or not? Or it could be continuous, like how much they'll repay."),
            ('A', "Mathematically, we often write this as Y equals f of X. X is your features, Y is your target, and f is the function connecting them."),
            ('B', "And that function f? That's your model. The thing you're trying to learn from data."),
            ('A', "A model is the learned representation. The mathematical function that maps inputs to outputs. Decision trees, neural networks, random forests, these are all types of models."),
            ('B', "Each model type has different strengths. Some are more interpretable, some handle complex patterns better, some need less data."),
            ('A', "One more key term. Training example. Also called a sample, instance, observation, or data point."),
            ('B', "It's simply one row in your dataset. One customer record. One email. One image. One complete set of features and, in supervised learning, one target value."),
            ('A', "If you have ten thousand customers, you have ten thousand training examples. The model learns from all of them to find general patterns."),
            ('B', "And that's really the goal. Learn patterns from training examples that generalize to new, unseen data. That's Machine Learning in essence."),
        ]
    },

    'definitions-statistics': {
        'title': 'Statistics Fundamentals - Complete Guide',
        'section': 'definitions',
        'dialogue': [
            ('A', "Statistics! This is the mathematical foundation of everything we do in data science. Let's really understand these concepts."),
            ('B', "And the good news is, the core ideas aren't complicated once you see the intuition behind them."),
            ('A', "Let's start with measures of central tendency. These answer the question: what's a typical or representative value in my data?"),
            ('B', "The mean is the most familiar. The average. You add up all your values and divide by the count. Simple arithmetic."),
            ('A', "If you have values ten, twenty, and thirty, the mean is ten plus twenty plus thirty, divided by three, equals twenty."),
            ('B', "The mean is great because it uses all your data. Every value contributes to the result."),
            ('A', "But here's the crucial weakness. The mean is sensitive to outliers. Extreme values can pull it dramatically."),
            ('B', "Imagine calculating average salary at a small company. Most employees make around fifty thousand. But the CEO makes ten million."),
            ('A', "Suddenly your average salary might be five hundred thousand. That doesn't represent what a typical employee makes at all!"),
            ('B', "One extreme value completely distorted the picture. This is when you need the median instead."),
            ('A', "The median is the middle value when you sort your data. Half the values are above it, half below."),
            ('B', "In that salary example, if you have fifty employees, you sort all salaries and pick the one in the middle. Maybe it's still around fifty thousand."),
            ('A', "The CEO's ten million salary only counts as one data point. It doesn't pull the median nearly as much."),
            ('B', "This is why news reports about income often use median instead of mean. It better represents the typical person."),
            ('A', "For skewed distributions, median is usually more informative. For symmetric distributions, mean and median are close anyway."),
            ('B', "Then there's the mode. Simply the most frequently occurring value. The value that appears the most."),
            ('A', "If your data is: one, two, two, three, three, three, four. The mode is three because it appears three times."),
            ('B', "Mode is especially useful for categorical data where mean and median don't make sense. What's the most common shirt size? That's the mode."),
            ('A', "Now let's talk about measures of spread. Central tendency tells you about typical values, but spread tells you how varied your data is."),
            ('B', "Variance is the fundamental measure. It quantifies how far, on average, each data point is from the mean."),
            ('A', "The calculation is: for each value, subtract the mean and square the result. Then average all those squared differences."),
            ('B', "Why square? Two reasons. First, it makes negative and positive differences both positive. They don't cancel out."),
            ('A', "Second, squaring penalizes large deviations more than small ones. A point far from the mean contributes more to variance."),
            ('B', "Let's do a quick example. Data: two, four, six. Mean is four. Now, two minus four is negative two, squared is four."),
            ('A', "Four minus four is zero, squared is zero. Six minus four is two, squared is four. Sum is eight, divide by three, variance is about two point six seven."),
            ('B', "The problem with variance is the units are squared. If you're measuring height in centimeters, variance is in centimeters squared. Weird to interpret."),
            ('A', "So we take the square root and get standard deviation. Same units as the original data. Much more intuitive."),
            ('B', "A standard deviation of five centimeters means data points typically vary about five centimeters from the mean."),
            ('A', "In a normal distribution, about sixty-eight percent of data falls within one standard deviation of the mean."),
            ('B', "And about ninety-five percent falls within two standard deviations. This gives you a sense of the spread."),
            ('A', "Now, correlation. This measures the linear relationship between two variables. Super important concept."),
            ('B', "Correlation ranges from negative one to positive one. The sign tells you the direction, the magnitude tells you the strength."),
            ('A', "Positive correlation means as one variable increases, the other tends to increase too. Height and weight for example."),
            ('B', "Negative correlation means as one increases, the other tends to decrease. Study time and error rate maybe."),
            ('A', "Zero correlation means no linear relationship. But be careful! This doesn't mean no relationship at all."),
            ('B', "There could be a curved relationship. A U-shape. Anything non-linear. Correlation only detects straight-line patterns."),
            ('A', "Correlation of point nine is strong positive. Point three is weak positive. Negative point seven is moderately strong negative."),
            ('B', "Last key concept: population versus sample. The population is the entire group you're interested in."),
            ('A', "All customers ever. All voters in a country. All stars in the galaxy. Usually too large or impossible to measure completely."),
            ('B', "So we take a sample. A subset. We measure the sample and use statistics to make inferences about the population."),
            ('A', "This is the heart of statistical inference. Using what we can observe to learn about what we can't fully observe."),
            ('B', "Sample size matters. Larger samples generally give more reliable estimates of population parameters."),
        ]
    },

    'definitions-preprocessing': {
        'title': 'Data Preprocessing - Complete Guide',
        'section': 'definitions',
        'dialogue': [
            ('A', "Data preprocessing. This is where data scientists spend the majority of their time. Some say up to eighty percent!"),
            ('B', "And it makes sense. Your model is only as good as the data you feed it. Garbage in, garbage out, as they say."),
            ('A', "Let's start with EDA. Exploratory Data Analysis. This should be your first step before any modeling."),
            ('B', "EDA is about getting to know your data intimately. Understanding its characteristics, quirks, and potential issues."),
            ('A', "What are the distributions of your variables? Are they normal, skewed, bimodal? Visualize them with histograms."),
            ('B', "Are there missing values? How many? Are they random or is there a pattern? This affects how you handle them."),
            ('A', "What are the correlations between variables? Are any highly correlated, suggesting redundancy?"),
            ('B', "Are there outliers? Extreme values that might be errors or might be genuine but unusual observations?"),
            ('A', "Don't skip EDA! I've seen people jump straight to modeling and then wonder why their results are garbage."),
            ('B', "Now let's talk encoding. Most machine learning algorithms work with numbers, not categories."),
            ('A', "One-Hot Encoding is the most common approach for categorical variables. Here's how it works."),
            ('B', "Say you have a color column with values red, blue, green. You create three new binary columns."),
            ('A', "Is Red, Is Blue, Is Green. Each one is zero or one. If something is red, it gets one in Is Red, zeros elsewhere."),
            ('B', "The big advantage: no false ordering. The algorithm doesn't think green is greater than blue is greater than red."),
            ('A', "The downside: many categories mean many new columns. A column with a hundred unique values becomes a hundred columns."),
            ('B', "Label Encoding is simpler. You just assign numbers. Red equals zero, blue equals one, green equals two."),
            ('A', "But now there's an implied order. The algorithm might think green is greater than blue. For some algorithms that's fine."),
            ('B', "Tree-based models like Random Forest handle label encoding well. Linear models can be misled by the false ordering."),
            ('A', "Scaling is crucial for many algorithms. It's about getting features onto similar numerical ranges."),
            ('B', "Imagine you have age, ranging from zero to one hundred, and income, ranging from zero to a million."),
            ('A', "Income has much bigger numbers. In distance-based algorithms like KNN, income would dominate completely."),
            ('B', "The distance would be almost entirely determined by income differences, ignoring age almost entirely."),
            ('A', "StandardScaler is one solution. It transforms data to have mean zero and standard deviation one."),
            ('B', "The formula is simple. Take your value, subtract the mean, divide by the standard deviation."),
            ('A', "A scaled value of two means that original value was two standard deviations above the mean."),
            ('B', "MinMaxScaler is another option. It squishes everything into the zero to one range."),
            ('A', "Formula: value minus minimum, divided by maximum minus minimum. The minimum becomes zero, maximum becomes one."),
            ('B', "This is good when you need bounded values, but it's more sensitive to outliers than StandardScaler."),
            ('A', "Now, data leakage. This is critical. This is one of the most common and dangerous mistakes."),
            ('B', "Data leakage happens when information from your test set influences your training process."),
            ('A', "Classic example: you scale your entire dataset, then split into training and test sets."),
            ('B', "Seems innocent, right? But the scaling was calculated using all data, including test data."),
            ('A', "Your model has indirectly seen information from the test set. Your evaluation will be optimistically biased."),
            ('B', "The correct approach: split first, then fit your scaler on training data only, then transform both sets."),
            ('A', "The scaler learns parameters from training data, applies those same parameters to test data."),
            ('B', "Any preprocessing that learns from data needs this treatment. Imputation, feature selection, everything."),
            ('A', "Speaking of imputation, that's how we handle missing values. Real data is messy and often has gaps."),
            ('B', "Simple imputation replaces missing values with the mean, median, or mode of that column."),
            ('A', "More sophisticated methods use other features to predict the missing value. Model-based imputation."),
            ('B', "But always think about why data is missing. Sometimes missingness itself is informative!"),
            ('A', "A missing income field might indicate unemployment. You might want to encode that information, not just fill it in."),
        ]
    },

    'definitions-evaluation': {
        'title': 'Model Evaluation - Complete Guide',
        'section': 'definitions',
        'dialogue': [
            ('A', "Model evaluation! You've built a model, but how do you know if it's actually any good? Let's dive deep into this."),
            ('B', "And there are so many metrics to choose from. The key is understanding what each measures and when to use which one."),
            ('A', "Let's start with the confusion matrix. Despite the name, it's actually quite simple once you break it down."),
            ('B', "For binary classification, positive versus negative, you have four possible outcomes. Let's go through each."),
            ('A', "True Positive, or TP. You predicted positive, and it actually was positive. You were right."),
            ('B', "Think: you said this email is spam, and it really was spam. Correct positive prediction."),
            ('A', "True Negative, TN. You predicted negative, it was negative. Also correct."),
            ('B', "You said this email is not spam, and it really wasn't. Correct negative prediction."),
            ('A', "False Positive, FP. You predicted positive, but it was actually negative. You were wrong."),
            ('B', "You said spam, but it was actually a legitimate email. A false alarm. Also called Type One error."),
            ('A', "Type One error is like crying wolf. Raising an alarm when there's no actual threat."),
            ('B', "False Negative, FN. You predicted negative, but it was actually positive. You missed it."),
            ('A', "You said not spam, but it actually was spam. It slipped through. Type Two error."),
            ('B', "Type Two error is missing something real. Failing to detect something that's actually there."),
            ('A', "Remember: Type One is false alarm, Type Two is missed detection. One false, two missed."),
            ('B', "Now, accuracy. The most intuitive metric. What fraction of all predictions were correct?"),
            ('A', "Formula: TP plus TN, divided by total predictions. Correct divided by total."),
            ('B', "If you got eighty-five out of a hundred predictions right, that's eighty-five percent accuracy."),
            ('A', "Sounds great, right? But accuracy has a huge problem with imbalanced data."),
            ('B', "Imagine fraud detection. Only one percent of transactions are fraud. Ninety-nine percent are legitimate."),
            ('A', "If your model just predicts not fraud for everything, it gets ninety-nine percent accuracy!"),
            ('B', "But it catches exactly zero frauds. Completely useless for its actual purpose."),
            ('A', "This is why we need precision and recall. They give a more nuanced picture."),
            ('B', "Precision asks: of all things I predicted positive, how many were actually positive?"),
            ('A', "Formula: TP divided by TP plus FP. Correct positives over all positive predictions."),
            ('B', "High precision means when you say positive, you're usually right. Few false alarms."),
            ('A', "Recall asks: of all actual positives, how many did I correctly identify?"),
            ('B', "Formula: TP divided by TP plus FN. Correct positives over all actual positives."),
            ('A', "High recall means you're catching most of the positives. Not missing many."),
            ('B', "Here's the key insight: there's often a trade-off between precision and recall."),
            ('A', "If you make your model more aggressive, calling more things positive, you might catch more true positives."),
            ('B', "But you'll also have more false positives. Recall goes up, precision goes down."),
            ('A', "If you make it more conservative, only flagging obvious cases, precision goes up but recall drops."),
            ('B', "Which to prioritize depends on the problem. For disease detection, you want high recall."),
            ('A', "Better to have some false alarms than to miss actual diseases. The cost of missing is too high."),
            ('B', "For a spam filter, you might prioritize precision. False positives are annoying, marking real emails as spam."),
            ('A', "F1 score tries to balance both. It's the harmonic mean of precision and recall."),
            ('B', "Formula: two times precision times recall, divided by precision plus recall."),
            ('A', "The harmonic mean has a nice property: if either precision or recall is very low, F1 will be low."),
            ('B', "You can't game it by just maximizing one and ignoring the other. Both matter."),
            ('A', "Specificity is like recall but for negatives. Of all actual negatives, how many did we correctly identify?"),
            ('B', "Formula: TN divided by TN plus FP. Useful when correctly identifying negatives matters."),
            ('A', "Finally, cross-validation. Not a metric exactly, but a technique for more robust evaluation."),
            ('B', "Instead of one train-test split, you do multiple. In five-fold cross-validation, you split data into five parts."),
            ('A', "Train on four parts, test on one. Repeat five times, each part being the test set once."),
            ('B', "Average the results. This gives a more reliable estimate of model performance than a single split."),
            ('A', "It's less dependent on the random chance of which examples ended up in training versus test."),
        ]
    },

    'definitions-algorithms': {
        'title': 'ML Algorithms - Complete Guide',
        'section': 'definitions',
        'dialogue': [
            ('A', "Machine Learning algorithms! These are the different methods for actually learning from data. Each has its own personality."),
            ('B', "Understanding the intuition helps you choose the right tool. Let's go through the major ones."),
            ('A', "Linear Regression. The classic. The grandfather of many techniques. Predicts a continuous output."),
            ('B', "The idea: the output is a weighted sum of inputs. Y equals a plus b times X."),
            ('A', "In higher dimensions with multiple features, you're finding the best hyperplane through your data."),
            ('B', "Best meaning it minimizes the prediction errors. Usually measured by squared error."),
            ('A', "It's simple, fast, and surprisingly effective for many problems. Don't underestimate it."),
            ('B', "And it's highly interpretable. Each coefficient tells you how much that feature affects the output."),
            ('A', "Logistic Regression. Despite the name, it's for classification, not regression. Confusing, I know."),
            ('B', "It uses the sigmoid function to squish outputs between zero and one. So they're like probabilities."),
            ('A', "The output might be point seven three. That means seventy-three percent confidence it's positive."),
            ('B', "Then you pick a threshold, usually point five, to make the final binary decision."),
            ('A', "Decision Trees. These are beautiful because they're so interpretable."),
            ('B', "The model is literally a tree of if-then questions. Is age greater than thirty? If yes, go left."),
            ('A', "Is income greater than fifty thousand? If yes, go left again. Keep going until you reach a leaf."),
            ('B', "Each leaf gives a prediction. You can literally read the decision logic out loud."),
            ('A', "This makes decision trees great when you need to explain your model to non-technical stakeholders."),
            ('B', "The downside: single trees tend to overfit. They can memorize the training data too specifically."),
            ('A', "That's where Random Forest comes in. An ensemble of many decision trees."),
            ('B', "Each tree is trained on a random subset of the data and features. Different trees learn different patterns."),
            ('A', "For prediction, all trees vote. Classification: majority class wins. Regression: average the predictions."),
            ('B', "This is called bagging. Bootstrap Aggregating. It makes the model much more robust."),
            ('A', "Random Forests are often one of the best out-of-the-box performers. Great default choice."),
            ('B', "Support Vector Machines, or SVMs. These find the hyperplane that best separates classes."),
            ('A', "But not just any separating hyperplane. The one with maximum margin. The biggest gap between classes."),
            ('B', "The intuition: you want as much buffer as possible. More margin, more confidence in new predictions."),
            ('A', "The data points closest to the boundary are called support vectors. They define the boundary."),
            ('B', "SVMs can also use kernel tricks to handle non-linear boundaries. Very powerful."),
            ('A', "Naive Bayes. A probabilistic classifier based on Bayes theorem."),
            ('B', "It calculates the probability of each class given the features you observe."),
            ('A', "The naive part is it assumes all features are independent of each other. Which is rarely true!"),
            ('B', "But surprisingly, even with this wrong assumption, Naive Bayes often works well in practice."),
            ('A', "It's especially popular for text classification. Fast training, fast prediction."),
            ('B', "K-Nearest Neighbors, or KNN. This one is different. It doesn't really learn a model."),
            ('A', "Instead, it stores all the training data. For a new point, it finds the K closest training points."),
            ('B', "Whatever class is most common among those neighbors is the prediction. Simple majority vote."),
            ('A', "K equals five means look at the five nearest points. K equals one just uses the single closest."),
            ('B', "Simple and intuitive, but can be slow for prediction with big datasets. And scaling is critical."),
            ('A', "One key thing: tree-based models like decision trees and random forests don't need feature scaling."),
            ('B', "But distance-based models like KNN and SVM definitely do. The distances would be distorted otherwise."),
        ]
    },

    'definitions-optimization': {
        'title': 'Model Optimization - Complete Guide',
        'section': 'definitions',
        'dialogue': [
            ('A', "Model optimization. So you have a model that works. How do you make it work better? Let's explore."),
            ('B', "There's a lot here. Hyperparameters, overfitting, underfitting. These concepts are crucial."),
            ('A', "First, let's distinguish parameters from hyperparameters. This confuses a lot of people."),
            ('B', "Parameters are values the algorithm learns from data during training. You don't set them."),
            ('A', "The weights in a neural network. The split points in a decision tree. The coefficients in linear regression."),
            ('B', "These are discovered by the learning algorithm as it processes your training data."),
            ('A', "Hyperparameters are different. You, the human, set these before training starts."),
            ('B', "Examples: the learning rate in gradient descent. How fast do we take steps toward the solution?"),
            ('A', "The maximum depth of a decision tree. How many levels of questions do we allow?"),
            ('B', "The number of trees in a random forest. K in KNN. The regularization strength."),
            ('A', "These choices affect how the algorithm learns, but they're not learned from data."),
            ('B', "Finding good hyperparameters is part science, part art. Grid search is one systematic approach."),
            ('A', "With grid search, you define values to try for each hyperparameter. Maybe max depth: three, five, seven, ten."),
            ('B', "And number of trees: fifty, one hundred, two hundred. That's twelve combinations."),
            ('A', "You train a model for each combination, evaluate with cross-validation, pick the best."),
            ('B', "Thorough but can be slow. The combinations explode quickly with more hyperparameters."),
            ('A', "Random search is an alternative. Randomly sample combinations instead of trying all."),
            ('B', "Often finds good hyperparameters faster because you're not wasting time on bad regions."),
            ('A', "Now, overfitting and underfitting. These are the two enemies of good machine learning."),
            ('B', "Overfitting is when your model learns the training data too well. It memorizes instead of generalizing."),
            ('A', "Classic sign: training accuracy is ninety-nine percent, but test accuracy is sixty percent. Huge gap."),
            ('B', "The model learned the noise and quirks specific to training data. Things that won't repeat."),
            ('A', "Causes: model is too complex for the amount of data. Or you trained for too many iterations."),
            ('B', "A decision tree with no limits might grow a unique path for almost every training example. Pure memorization."),
            ('A', "Solutions: simplify the model, get more data, use regularization, or stop training earlier."),
            ('B', "Underfitting is the opposite. The model is too simple to capture the underlying patterns."),
            ('A', "Both training and test accuracy are low. The model can't even fit the training data well."),
            ('B', "It's like trying to fit a straight line to data that's clearly curved. The model lacks capacity."),
            ('A', "Solutions: use a more complex model, add more features, train longer, or reduce regularization."),
            ('B', "The sweet spot is between overfitting and underfitting. Complex enough to learn patterns, simple enough to generalize."),
            ('A', "SMOTE. Synthetic Minority Over-sampling Technique. This addresses class imbalance."),
            ('B', "When one class is way more common. Like fraud: maybe one percent fraud, ninety-nine percent normal."),
            ('A', "If you train naively, the model might just predict normal every time. Easy path to high accuracy."),
            ('B', "SMOTE creates synthetic examples of the minority class. It finds minority points and their neighbors."),
            ('A', "It creates new synthetic points along the lines connecting them. Essentially interpolating."),
            ('B', "This balances the dataset so the model learns to recognize the minority class better."),
            ('A', "Not perfect, you're creating fake data. But often significantly improves minority class detection."),
            ('B', "Use with caution though. Apply only to training data, never to test data. Otherwise you're cheating."),
        ]
    },

    # ========== FORMULAS SECTION ==========
    'formulas-all': {
        'title': 'All Formulas - Detailed Walkthrough',
        'section': 'formulas',
        'dialogue': [
            ('A', "Let's go through all the key formulas you need to know. With worked examples for each one."),
            ('B', "We'll make sure you understand the intuition, not just memorize symbols."),
            ('A', "Starting with the mean. Sum of all values divided by count. The most basic measure of central tendency."),
            ('B', "Example: values are ten, twenty, thirty, forty, fifty. Sum is one fifty. Divide by five. Mean is thirty."),
            ('A', "The mean uses all data points, which is good. But it's sensitive to outliers, which can be bad."),
            ('B', "Variance measures spread. Sum of squared differences from the mean, divided by n."),
            ('A', "Let's work through it. Data: two, four, six. Mean is four."),
            ('B', "Two minus four is negative two. Squared is four. Four minus four is zero. Squared is zero. Six minus four is two. Squared is four."),
            ('A', "Sum the squares: four plus zero plus four equals eight. Divide by three. Variance is two point six seven."),
            ('B', "Why square? Makes negatives positive, and penalizes big deviations more than small ones."),
            ('A', "Standard deviation is just the square root of variance. So square root of two point six seven is about one point six three."),
            ('B', "Now in the same units as your data. Much easier to interpret."),
            ('A', "Accuracy formula: TP plus TN, all divided by total. How many did you get right overall?"),
            ('B', "Example: TP is eighty, TN is ninety, FP is ten, FN is twenty. Total is two hundred."),
            ('A', "Eighty plus ninety is one seventy. Divide by two hundred. That's eighty-five percent accuracy."),
            ('B', "Precision formula: TP divided by TP plus FP. Of positive predictions, how many were right?"),
            ('A', "With TP eighty and FP twenty: eighty over one hundred equals eighty percent precision."),
            ('B', "Recall formula: TP divided by TP plus FN. Of actual positives, how many did you catch?"),
            ('A', "With TP eighty and FN twenty: eighty over one hundred equals eighty percent recall."),
            ('B', "F1 Score: two times precision times recall, divided by precision plus recall. Harmonic mean."),
            ('A', "Example: precision point eight, recall point six. Two times point four eight is point nine six."),
            ('B', "Point eight plus point six is one point four. Point nine six over one point four equals about point six nine."),
            ('A', "Notice F1 is closer to the lower of precision and recall. That's the harmonic mean effect."),
            ('B', "Specificity: TN divided by TN plus FP. Of actual negatives, how many correctly identified?"),
            ('A', "With TN ninety, FP ten: ninety over one hundred equals ninety percent specificity."),
            ('B', "StandardScaler formula: X minus mean, divided by standard deviation. Centers and scales data."),
            ('A', "Example: X is seventy, mean is fifty, standard deviation is ten. Seventy minus fifty is twenty."),
            ('B', "Twenty divided by ten equals two. That two means the value is two standard deviations above the mean."),
            ('A', "MinMaxScaler formula: X minus min, divided by max minus min. Scales to zero-one range."),
            ('B', "Example: X is thirty, min is ten, max is fifty. Thirty minus ten is twenty. Fifty minus ten is forty."),
            ('A', "Twenty over forty equals point five. Makes sense, thirty is exactly in the middle of ten to fifty."),
            ('B', "Pearson correlation: covariance of X and Y, divided by product of their standard deviations."),
            ('A', "Example: covariance is fifteen, standard deviation of X is five, of Y is three."),
            ('B', "Fifteen divided by five times three, which is fifteen. Fifteen over fifteen equals one. Perfect positive correlation."),
            ('A', "That means as X increases, Y increases proportionally. Perfect linear relationship."),
            ('B', "Practice these formulas until they're automatic. Know what each measures and when to use it."),
        ]
    },

    # ========== PRACTICE SECTION ==========
    'practice-confusion-matrix': {
        'title': 'Confusion Matrix Practice',
        'section': 'practice',
        'dialogue': [
            ('A', "Let's practice confusion matrix calculations. This is a common exam question. The key is being systematic."),
            ('B', "We'll walk through a complete problem step by step."),
            ('A', "Scenario: a model predicts disease outcomes. The confusion matrix shows: actual positive row has forty-five predicted positive, five predicted negative."),
            ('B', "Actual negative row has fifteen predicted positive, one thirty-five predicted negative."),
            ('A', "First step: identify TP, TN, FP, FN from the matrix layout."),
            ('B', "True Positive: actual positive AND predicted positive. Top left: forty-five."),
            ('A', "True Negative: actual negative AND predicted negative. Bottom right: one thirty-five."),
            ('B', "False Positive: actual negative BUT predicted positive. Bottom left: fifteen."),
            ('A', "False Negative: actual positive BUT predicted negative. Top right: five."),
            ('B', "Great. TP is forty-five, TN is one thirty-five, FP is fifteen, FN is five. Total is two hundred."),
            ('A', "Accuracy: TP plus TN over total. Forty-five plus one thirty-five is one eighty. Divide by two hundred. Ninety percent."),
            ('B', "Precision: TP over TP plus FP. Forty-five over sixty. Seventy-five percent."),
            ('A', "Recall: TP over TP plus FN. Forty-five over fifty. Ninety percent."),
            ('B', "F1: two times point seven five times point nine, over point seven five plus point nine."),
            ('A', "Two times point six seven five is one point three five. Over one point six five. About eighty-two percent."),
            ('B', "Now the insight question: is this model good for disease detection?"),
            ('A', "Ninety percent recall means we catch ninety percent of diseases. Only miss ten percent. Good!"),
            ('B', "But seventy-five percent precision means twenty-five percent of positive predictions are false alarms."),
            ('A', "In medical contexts, high recall is usually more important. Better false alarms than missed diseases."),
            ('B', "So this model is reasonably good. Could improve precision if false positives are costly."),
        ]
    },

    'practice-statistics': {
        'title': 'Statistics Practice',
        'section': 'practice',
        'dialogue': [
            ('A', "Let's practice statistics calculations. Being systematic is key."),
            ('B', "Dataset: twelve, fifteen, eighteen, twenty-two, twenty-five, twenty-eight, thirty. Seven values."),
            ('A', "Mean: add them all. Twelve plus fifteen is twenty-seven. Plus eighteen is forty-five."),
            ('B', "Plus twenty-two is sixty-seven. Plus twenty-five is ninety-two. Plus twenty-eight is one twenty. Plus thirty is one fifty."),
            ('A', "One fifty divided by seven values. That's about twenty-one point four three."),
            ('B', "Median: data is already sorted. With seven values, middle is the fourth one."),
            ('A', "Count: twelve is first, fifteen second, eighteen third, twenty-two fourth. Median is twenty-two."),
            ('B', "Mean and median are close. Twenty-one point four three versus twenty-two. Suggests symmetric distribution."),
            ('A', "Range: max minus min. Thirty minus twelve equals eighteen."),
            ('B', "Always double-check: does the answer make sense? Mean should be between min and max."),
        ]
    },

    'practice-scaling': {
        'title': 'Scaling Practice',
        'section': 'practice',
        'dialogue': [
            ('A', "Scaling problems. These are actually quite straightforward once you know the formulas."),
            ('B', "Data: one hundred, two hundred, three hundred, four hundred, five hundred. Scale three hundred."),
            ('A', "MinMaxScaler: X minus min over max minus min. Min is one hundred, max is five hundred."),
            ('B', "Three hundred minus one hundred is two hundred. Five hundred minus one hundred is four hundred."),
            ('A', "Two hundred over four hundred equals point five. Makes sense, three hundred is exactly in the middle."),
            ('B', "StandardScaler: need mean and standard deviation. Mean of this data is three hundred."),
            ('A', "Standard deviation is about one forty-one point four."),
            ('B', "StandardScaler: X minus mean over standard deviation. Three hundred minus three hundred is zero."),
            ('A', "Zero divided by anything is zero. Scaled value is exactly zero."),
            ('B', "This is correct! The mean value always standardizes to exactly zero. Good sanity check."),
        ]
    },

    'practice-imbalance': {
        'title': 'Class Imbalance Practice',
        'section': 'practice',
        'dialogue': [
            ('A', "Class imbalance. This shows why accuracy can be misleading."),
            ('B', "Scenario: nine fifty normal samples, fifty fraud samples. One thousand total."),
            ('A', "Imbalance ratio: nine fifty to fifty simplifies to nineteen to one. Or ninety-five percent vs five percent."),
            ('B', "Key question: if we predict everything as normal, what's the accuracy?"),
            ('A', "We get all nine fifty normal cases correct. All fifty fraud cases wrong."),
            ('B', "Nine fifty out of one thousand. That's ninety-five percent accuracy!"),
            ('A', "Sounds amazing. But think about what this model actually does."),
            ('B', "Precision for fraud: TP over TP plus FP. We predict zero fraud, so TP is zero. Zero percent."),
            ('A', "Recall for fraud: TP over TP plus FN. TP is zero, FN is fifty. Also zero percent."),
            ('B', "We catch exactly zero frauds. Completely useless for its purpose!"),
            ('A', "This is why with imbalanced data, use precision, recall, F1. Not just accuracy."),
            ('B', "Or use techniques like SMOTE to balance, or adjust class weights in your algorithm."),
        ]
    },

    # ========== QUICK REFERENCE SECTION ==========
    'quickref-all': {
        'title': 'Complete Quick Reference',
        'section': 'quickref',
        'dialogue': [
            ('A', "Quick reference time. Let's rapid-fire through all the key points."),
            ('B', "Confusion matrix: TP correct positive, TN correct negative, FP false alarm, FN missed."),
            ('A', "Type one is FP, false alarm. Type two is FN, missed it."),
            ('B', "Accuracy is TP plus TN over total. Don't trust it for imbalanced data!"),
            ('A', "Precision: TP over TP plus FP. When you predict positive, how often right?"),
            ('B', "Recall: TP over TP plus FN. Of actual positives, how many caught?"),
            ('A', "F1 is harmonic mean. Two PR over P plus R."),
            ('B', "Disease and fraud detection: prioritize recall. Don't miss cases."),
            ('A', "Spam filters: might prioritize precision. False positives annoy users."),
            ('B', "StandardScaler: mean zero, std one. X minus mean over sigma."),
            ('A', "MinMaxScaler: zero to one range. X minus min over max minus min."),
            ('B', "Always fit on training, transform both. Never fit on whole dataset. That's leakage!"),
            ('A', "Tree models don't need scaling. Distance models like KNN, SVM need it."),
            ('B', "One-hot encoding: binary columns, no false order. Label encoding: numbers, implies order."),
            ('A', "Overfitting: high train accuracy, low test. Model too complex."),
            ('B', "Underfitting: both train and test low. Model too simple."),
            ('A', "Cross-validation: split K ways, train K times, average. More robust."),
            ('B', "SMOTE for imbalance. Creates synthetic minority samples."),
            ('A', "Random Forest is bagging. Many trees vote. Reduces overfitting."),
            ('B', "Mean sensitive to outliers. Median robust. Use median for skewed data."),
            ('A', "Correlation zero means no linear relationship. Could still be non-linear!"),
            ('B', "Review these until they're automatic!"),
        ]
    },
}


async def generate_dialogue_audio(topic_id: str, output_path: Path = None) -> str:
    """Generate podcast audio for a topic with natural speech"""

    if topic_id not in PODCAST_SCRIPTS:
        raise ValueError(f"Unknown topic: {topic_id}")

    script = PODCAST_SCRIPTS[topic_id]
    dialogue = script['dialogue']

    # Create audio directory if needed
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        output_path = AUDIO_DIR / f"{topic_id}_podcast.mp3"

    temp_files = []
    concat_list = AUDIO_DIR / f"concat_{topic_id}.txt"

    try:
        for i, (speaker, text) in enumerate(dialogue):
            voice = HOST_A if speaker == 'A' else HOST_B
            temp_file = AUDIO_DIR / f"temp_{topic_id}_{i}.mp3"
            temp_files.append(temp_file)

            communicate = edge_tts.Communicate(
                text,
                voice,
                rate="-8%" if speaker == 'A' else "-5%",
                pitch="-10Hz" if speaker == 'A' else "+5Hz"
            )
            await communicate.save(str(temp_file))

        # Create ffmpeg concat file list
        with open(concat_list, 'w') as f:
            for temp_file in temp_files:
                f.write(f"file '{temp_file}'\n")

        # Use ffmpeg to properly concatenate with seekable output
        subprocess.run([
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', str(concat_list),
            '-c:a', 'libmp3lame', '-q:a', '2',
            str(output_path)
        ], capture_output=True, check=True)

        return str(output_path)

    finally:
        # Cleanup temp files
        for temp_file in temp_files:
            if temp_file.exists():
                temp_file.unlink()
        if concat_list.exists():
            concat_list.unlink()


def generate_topic_audio(topic_id: str) -> str:
    """Synchronous wrapper for audio generation"""
    return asyncio.run(generate_dialogue_audio(topic_id))


def get_available_topics() -> list:
    """Get list of topics with podcast scripts"""
    return [
        {'id': topic_id, 'title': data['title'], 'section': data.get('section', 'general')}
        for topic_id, data in PODCAST_SCRIPTS.items()
    ]


def get_topics_by_section(section: str) -> list:
    """Get topics for a specific section"""
    return [
        {'id': topic_id, 'title': data['title']}
        for topic_id, data in PODCAST_SCRIPTS.items()
        if data.get('section') == section
    ]


def audio_exists(topic_id: str) -> bool:
    """Check if audio file exists for topic"""
    audio_path = AUDIO_DIR / f"{topic_id}_podcast.mp3"
    return audio_path.exists()


def get_audio_path(topic_id: str) -> str:
    """Get the audio file path for a topic"""
    return f"/static/audio/{topic_id}_podcast.mp3"


def get_script(topic_id: str) -> dict:
    """Get the script for a topic"""
    return PODCAST_SCRIPTS.get(topic_id, {})


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        topic = sys.argv[1]
    else:
        topic = "definitions-ml-basics"

    print(f"Generating podcast for: {topic}")
    print(f"This may take a minute...")
    path = generate_topic_audio(topic)
    print(f"Audio saved to: {path}")
