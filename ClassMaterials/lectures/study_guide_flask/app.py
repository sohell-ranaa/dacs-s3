#!/usr/bin/env python3
"""
DACS Machine Learning Study Guide - Clean Flask Application
CT115-3-M Data Analytics in Cyber Security
Focus: Definitions, Formulas, Examples, Practice
"""

from flask import Flask, render_template, jsonify, send_from_directory
import os

app = Flask(__name__)

# Import podcast generator
from podcast_generator import (
    PODCAST_SCRIPTS,
    generate_topic_audio,
    get_available_topics,
    get_topics_by_section,
    audio_exists,
    get_audio_path,
    get_script,
    AUDIO_DIR
)
app.config['SECRET_KEY'] = 'dacs-study-guide-2024'

# ============================================
# KEY DEFINITIONS - Organized by Topic
# ============================================
DEFINITIONS = {
    'ml-basics': {
        'title': 'ML Basics',
        'terms': [
            {'term': 'Machine Learning', 'meaning': 'A field of AI that enables computers to learn from data without being explicitly programmed.', 'example': 'Spam filter that learns from examples of spam emails'},
            {'term': 'Supervised Learning', 'meaning': 'Learning from labeled data where both input (X) and output (Y) are provided. The model learns the function Y = f(X).', 'example': 'Classification (spam/not spam), Regression (price prediction)'},
            {'term': 'Unsupervised Learning', 'meaning': 'Learning from unlabeled data to find hidden patterns or structures.', 'example': 'Customer segmentation, anomaly detection'},
            {'term': 'Feature', 'meaning': 'An input variable (column) used to make predictions. Also called predictor, attribute, or independent variable.', 'example': 'Age, income, location in customer data'},
            {'term': 'Target', 'meaning': 'The output variable we want to predict. Also called label, ground truth, or dependent variable.', 'example': 'Whether customer will churn (Yes/No)'},
            {'term': 'Training Example', 'meaning': 'A single row of data containing features and (in supervised learning) the target value.', 'example': 'One customer record with all attributes'},
            {'term': 'Model', 'meaning': 'The function y = f(x) learned from training data that maps inputs to outputs.', 'example': 'Decision tree, neural network, SVM'},
        ]
    },
    'statistics': {
        'title': 'Statistics',
        'terms': [
            {'term': 'Mean', 'meaning': 'The average value. Sum all values and divide by count.', 'example': 'Mean of [2,4,6] = (2+4+6)/3 = 4'},
            {'term': 'Median', 'meaning': 'The middle value when data is sorted. Less affected by outliers than mean.', 'example': 'Median of [1,2,100] = 2 (not 34.3 like mean)'},
            {'term': 'Mode', 'meaning': 'The most frequently occurring value in a dataset.', 'example': 'Mode of [1,2,2,3,3,3] = 3'},
            {'term': 'Variance', 'meaning': 'Average of squared differences from the mean. Measures spread of data.', 'example': 'High variance = data is spread out'},
            {'term': 'Standard Deviation', 'meaning': 'Square root of variance. Same unit as the data, easier to interpret.', 'example': 'σ = 5 means typical values are within 5 units of mean'},
            {'term': 'Correlation', 'meaning': 'Measures linear relationship between two variables. Range: -1 to +1.', 'example': 'ρ = 0.9 means strong positive relationship'},
            {'term': 'Population', 'meaning': 'The entire group being studied.', 'example': 'All customers of a company'},
            {'term': 'Sample', 'meaning': 'A subset of the population used for analysis.', 'example': '1000 randomly selected customers'},
        ]
    },
    'preprocessing': {
        'title': 'Data Preprocessing',
        'terms': [
            {'term': 'EDA', 'meaning': 'Exploratory Data Analysis - examining data to summarize characteristics and find patterns.', 'example': 'Checking distributions, correlations, missing values'},
            {'term': 'One-Hot Encoding', 'meaning': 'Converting categorical variables to binary columns (0 or 1).', 'example': 'Color: Red→[1,0,0], Blue→[0,1,0], Green→[0,0,1]'},
            {'term': 'Label Encoding', 'meaning': 'Converting categories to numbers. May create false ordinal relationships.', 'example': 'Red=0, Blue=1, Green=2 (implies order)'},
            {'term': 'StandardScaler', 'meaning': 'Transforms data to have mean=0 and std=1.', 'example': 'x_scaled = (x - mean) / std'},
            {'term': 'MinMaxScaler', 'meaning': 'Transforms data to range [0, 1].', 'example': 'x_scaled = (x - min) / (max - min)'},
            {'term': 'Data Leakage', 'meaning': 'When information from test set influences training, causing overly optimistic results.', 'example': 'Scaling entire dataset before train/test split'},
            {'term': 'Imputation', 'meaning': 'Filling in missing values with estimated values.', 'example': 'Replace missing age with mean age'},
        ]
    },
    'evaluation': {
        'title': 'Model Evaluation',
        'terms': [
            {'term': 'True Positive (TP)', 'meaning': 'Model correctly predicted positive class.', 'example': 'Predicted fraud, actually was fraud'},
            {'term': 'True Negative (TN)', 'meaning': 'Model correctly predicted negative class.', 'example': 'Predicted not fraud, actually not fraud'},
            {'term': 'False Positive (FP)', 'meaning': 'Model incorrectly predicted positive. Type I Error.', 'example': 'Predicted fraud, but was not fraud (false alarm)'},
            {'term': 'False Negative (FN)', 'meaning': 'Model incorrectly predicted negative. Type II Error.', 'example': 'Predicted not fraud, but was fraud (missed)'},
            {'term': 'Precision', 'meaning': 'Of all positive predictions, how many were correct? TP/(TP+FP)', 'example': 'When we predict fraud, how often are we right?'},
            {'term': 'Recall', 'meaning': 'Of all actual positives, how many did we find? TP/(TP+FN)', 'example': 'Of all frauds, how many did we catch?'},
            {'term': 'F1 Score', 'meaning': 'Harmonic mean of precision and recall. Balances both metrics.', 'example': 'F1 = 2 × (P × R) / (P + R)'},
            {'term': 'Cross-Validation', 'meaning': 'Technique to evaluate model by training and testing on different data splits.', 'example': '5-fold CV: train on 4 parts, test on 1, repeat 5 times'},
        ]
    },
    'algorithms': {
        'title': 'Algorithms',
        'terms': [
            {'term': 'Linear Regression', 'meaning': 'Predicts continuous output as weighted sum of inputs: y = a + bx', 'example': 'Predicting house price from square footage'},
            {'term': 'Logistic Regression', 'meaning': 'Predicts probability of binary outcome using sigmoid function.', 'example': 'Probability of customer churn'},
            {'term': 'Decision Tree', 'meaning': 'Makes decisions by splitting data based on feature values. Easy to interpret.', 'example': 'If age > 30 AND income > 50k, then approve loan'},
            {'term': 'Random Forest', 'meaning': 'Ensemble of decision trees trained on random subsets. Bagging method.', 'example': '100 trees vote, majority wins'},
            {'term': 'SVM', 'meaning': 'Support Vector Machine finds hyperplane that best separates classes.', 'example': 'Points closest to boundary are support vectors'},
            {'term': 'Naive Bayes', 'meaning': 'Probabilistic classifier assuming features are independent.', 'example': 'Fast, good for text classification'},
            {'term': 'KNN', 'meaning': 'K-Nearest Neighbors classifies based on closest training examples.', 'example': 'K=5: look at 5 nearest points, majority class wins'},
        ]
    },
    'optimization': {
        'title': 'Optimization',
        'terms': [
            {'term': 'Hyperparameter', 'meaning': 'Settings configured before training (not learned from data).', 'example': 'Learning rate, max_depth, number of trees'},
            {'term': 'Parameter', 'meaning': 'Values learned during training.', 'example': 'Weights and biases in neural network'},
            {'term': 'Grid Search', 'meaning': 'Tests all combinations of hyperparameter values.', 'example': 'Try max_depth=[3,5,7] × n_estimators=[50,100,200]'},
            {'term': 'Overfitting', 'meaning': 'Model learns training data too well, fails on new data. High variance.', 'example': 'Train accuracy 99%, test accuracy 60%'},
            {'term': 'Underfitting', 'meaning': 'Model too simple to capture patterns. High bias.', 'example': 'Both train and test accuracy are low'},
            {'term': 'SMOTE', 'meaning': 'Synthetic Minority Over-sampling Technique - creates synthetic samples of minority class.', 'example': 'Balance 100 fraud vs 10000 normal cases'},
        ]
    }
}

# ============================================
# KEY FORMULAS with Examples
# ============================================
FORMULAS = [
    {
        'name': 'Mean (Average)',
        'latex': r'\bar{x} = \frac{\sum_{i=1}^{n} x_i}{n}',
        'description': 'Sum of all values divided by count',
        'example': {
            'question': 'Find mean of: 10, 20, 30, 40, 50',
            'solution': 'Mean = (10+20+30+40+50)/5 = 150/5 = 30'
        }
    },
    {
        'name': 'Variance',
        'latex': r'\sigma^2 = \frac{\sum_{i=1}^{n}(x_i - \bar{x})^2}{n}',
        'description': 'Average of squared differences from mean',
        'example': {
            'question': 'Find variance of: 2, 4, 6 (mean=4)',
            'solution': 'Variance = [(2-4)² + (4-4)² + (6-4)²]/3 = [4+0+4]/3 = 8/3 = 2.67'
        }
    },
    {
        'name': 'Standard Deviation',
        'latex': r'\sigma = \sqrt{\sigma^2}',
        'description': 'Square root of variance',
        'example': {
            'question': 'If variance = 16, find standard deviation',
            'solution': 'σ = √16 = 4'
        }
    },
    {
        'name': 'Accuracy',
        'latex': r'Accuracy = \frac{TP + TN}{TP + TN + FP + FN}',
        'description': 'Proportion of correct predictions',
        'example': {
            'question': 'TP=80, TN=90, FP=10, FN=20. Find accuracy.',
            'solution': 'Accuracy = (80+90)/(80+90+10+20) = 170/200 = 0.85 = 85%'
        }
    },
    {
        'name': 'Precision',
        'latex': r'Precision = \frac{TP}{TP + FP}',
        'description': 'Of positive predictions, how many correct?',
        'example': {
            'question': 'TP=80, FP=20. Find precision.',
            'solution': 'Precision = 80/(80+20) = 80/100 = 0.80 = 80%'
        }
    },
    {
        'name': 'Recall (Sensitivity)',
        'latex': r'Recall = \frac{TP}{TP + FN}',
        'description': 'Of actual positives, how many found?',
        'example': {
            'question': 'TP=80, FN=20. Find recall.',
            'solution': 'Recall = 80/(80+20) = 80/100 = 0.80 = 80%'
        }
    },
    {
        'name': 'F1 Score',
        'latex': r'F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}',
        'description': 'Harmonic mean of precision and recall',
        'example': {
            'question': 'Precision=0.8, Recall=0.6. Find F1.',
            'solution': 'F1 = 2×(0.8×0.6)/(0.8+0.6) = 2×0.48/1.4 = 0.96/1.4 = 0.686'
        }
    },
    {
        'name': 'Specificity',
        'latex': r'Specificity = \frac{TN}{TN + FP}',
        'description': 'Of actual negatives, how many correctly identified?',
        'example': {
            'question': 'TN=90, FP=10. Find specificity.',
            'solution': 'Specificity = 90/(90+10) = 90/100 = 0.90 = 90%'
        }
    },
    {
        'name': 'StandardScaler',
        'latex': r'x_{scaled} = \frac{x - \mu}{\sigma}',
        'description': 'Transform to mean=0, std=1',
        'example': {
            'question': 'x=70, mean=50, std=10. Standardize x.',
            'solution': 'x_scaled = (70-50)/10 = 20/10 = 2.0'
        }
    },
    {
        'name': 'MinMaxScaler',
        'latex': r'x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}',
        'description': 'Transform to range [0, 1]',
        'example': {
            'question': 'x=30, min=10, max=50. Scale x.',
            'solution': 'x_scaled = (30-10)/(50-10) = 20/40 = 0.5'
        }
    },
    {
        'name': 'Pearson Correlation',
        'latex': r'\rho = \frac{cov(X,Y)}{\sigma_X \cdot \sigma_Y}',
        'description': 'Linear relationship strength (-1 to +1)',
        'example': {
            'question': 'cov(X,Y)=15, σ_X=5, σ_Y=3. Find correlation.',
            'solution': 'ρ = 15/(5×3) = 15/15 = 1.0 (perfect positive)'
        }
    },
]

# ============================================
# PRACTICE PROBLEMS - Calculations
# ============================================
PRACTICE_PROBLEMS = [
    {
        'id': 1,
        'topic': 'Confusion Matrix',
        'question': '''A model predicts disease outcomes with the following confusion matrix:

        |  | Predicted + | Predicted - |
        |---|---|---|
        | Actual + | 45 | 5 |
        | Actual - | 15 | 135 |

        Calculate: (a) Accuracy (b) Precision (c) Recall (d) F1 Score''',
        'solution': '''From the matrix: TP=45, FN=5, FP=15, TN=135, Total=200

(a) Accuracy = (TP+TN)/(Total) = (45+135)/200 = 180/200 = 0.90 = 90%

(b) Precision = TP/(TP+FP) = 45/(45+15) = 45/60 = 0.75 = 75%

(c) Recall = TP/(TP+FN) = 45/(45+5) = 45/50 = 0.90 = 90%

(d) F1 = 2×(P×R)/(P+R) = 2×(0.75×0.90)/(0.75+0.90) = 2×0.675/1.65 = 0.818 = 81.8%''',
        'key_insight': 'High recall (90%) means we catch most diseases. Lower precision (75%) means some false alarms. For disease detection, high recall is usually preferred.'
    },
    {
        'id': 2,
        'topic': 'Statistics',
        'question': '''Dataset: 12, 15, 18, 22, 25, 28, 30

Calculate: (a) Mean (b) Median (c) Range''',
        'solution': '''(a) Mean = Sum/Count = (12+15+18+22+25+28+30)/7 = 150/7 = 21.43

(b) Median = Middle value (already sorted)
    7 values → middle is 4th value = 22

(c) Range = Max - Min = 30 - 12 = 18''',
        'key_insight': 'Mean (21.43) and median (22) are close, suggesting roughly symmetric distribution.'
    },
    {
        'id': 3,
        'topic': 'Scaling',
        'question': '''Feature values: [100, 200, 300, 400, 500]

Scale the value x=300 using:
(a) MinMaxScaler
(b) StandardScaler (mean=300, std=141.4)''',
        'solution': '''(a) MinMaxScaler:
    min=100, max=500
    x_scaled = (300-100)/(500-100) = 200/400 = 0.5

(b) StandardScaler:
    x_scaled = (300-300)/141.4 = 0/141.4 = 0.0''',
        'key_insight': 'After StandardScaler, x=300 (the mean) becomes 0. After MinMaxScaler, x=300 (middle value) becomes 0.5.'
    },
    {
        'id': 4,
        'topic': 'Class Imbalance',
        'question': '''Training set has 950 normal samples and 50 fraud samples.

(a) What is the class imbalance ratio?
(b) If model predicts ALL as "normal", what is its accuracy?
(c) Why is accuracy misleading here?''',
        'solution': '''(a) Imbalance ratio = 950:50 = 19:1 (or 95% vs 5%)

(b) If predict all as normal:
    - Correct: 950 normal predictions
    - Wrong: 50 fraud missed
    - Accuracy = 950/1000 = 95%

(c) 95% accuracy sounds great, but:
    - Precision for fraud = 0%
    - Recall for fraud = 0%
    - We catch ZERO frauds!''',
        'key_insight': 'With imbalanced data, use precision, recall, and F1 instead of accuracy.'
    },
    {
        'id': 5,
        'topic': 'Model Comparison',
        'question': '''Two models tested:
        - Model A: Accuracy=92%, Precision=85%, Recall=75%
        - Model B: Accuracy=88%, Precision=70%, Recall=95%

Which model is better for disease detection?''',
        'solution': '''For disease detection, missing a disease (False Negative) is dangerous.

Model A: Recall=75% → misses 25% of diseases
Model B: Recall=95% → misses only 5% of diseases

Model B is better because:
- Higher recall (catches more diseases)
- Even though precision is lower (more false alarms),
  it's better to investigate a false alarm than miss a disease

F1 scores:
- A: 2×(0.85×0.75)/(0.85+0.75) = 0.80
- B: 2×(0.70×0.95)/(0.70+0.95) = 0.81''',
        'key_insight': 'Choose metric based on business need: Recall for disease/fraud detection, Precision for spam filters.'
    },
]

# ============================================
# TOPICS for Navigation
# ============================================
TOPICS = [
    {'id': 1, 'slug': 'ml-fundamentals', 'title': 'ML Fundamentals', 'color': '#2563eb'},
    {'id': 2, 'slug': 'statistics', 'title': 'Statistics', 'color': '#10b981'},
    {'id': 3, 'slug': 'ml-pipeline', 'title': 'ML Pipeline', 'color': '#f59e0b'},
    {'id': 4, 'slug': 'preprocessing', 'title': 'Preprocessing', 'color': '#8b5cf6'},
    {'id': 5, 'slug': 'evaluation', 'title': 'Evaluation', 'color': '#ef4444'},
    {'id': 6, 'slug': 'algorithms', 'title': 'Algorithms', 'color': '#059669'},
    {'id': 7, 'slug': 'optimization', 'title': 'Optimization', 'color': '#6366f1'},
    {'id': 8, 'slug': 'selection', 'title': 'Model Selection', 'color': '#0891b2'},
]

# ============================================
# QUIZ DATA
# ============================================
QUIZ_DATA = {
    'evaluation': [
        {'q': 'What is Precision?', 'opts': ['TP/(TP+FN)', 'TP/(TP+FP)', 'TN/(TN+FP)', 'TN/(TN+FN)'], 'ans': 1},
        {'q': 'What is Recall?', 'opts': ['TP/(TP+FN)', 'TP/(TP+FP)', 'TN/(TN+FP)', 'TN/(TN+FN)'], 'ans': 0},
        {'q': 'Type I Error is:', 'opts': ['False Negative', 'False Positive', 'True Positive', 'True Negative'], 'ans': 1},
        {'q': 'Type II Error is:', 'opts': ['False Negative', 'False Positive', 'True Positive', 'True Negative'], 'ans': 0},
        {'q': 'For disease detection, prioritize:', 'opts': ['Precision', 'Recall', 'Accuracy', 'Specificity'], 'ans': 1},
        {'q': 'F1 Score is the ____ mean of P and R:', 'opts': ['Arithmetic', 'Geometric', 'Harmonic', 'Weighted'], 'ans': 2},
    ],
    'statistics': [
        {'q': 'For positively skewed data:', 'opts': ['Mean > Median > Mode', 'Mode > Median > Mean', 'Mean = Median = Mode', 'Median > Mean > Mode'], 'ans': 0},
        {'q': 'Correlation = 0 means:', 'opts': ['Variables are independent', 'No LINEAR relationship', 'Perfect negative', 'Strong positive'], 'ans': 1},
        {'q': 'Standard deviation is:', 'opts': ['Variance squared', 'Square root of variance', 'Same as variance', 'Mean squared'], 'ans': 1},
        {'q': 'Sample is:', 'opts': ['Entire population', 'Subset of population', 'Always random', 'Always large'], 'ans': 1},
    ],
    'preprocessing': [
        {'q': 'To prevent data leakage when scaling:', 'opts': ['Scale before split', 'Fit on test set', 'Fit on train, transform both', 'Never scale'], 'ans': 2},
        {'q': 'StandardScaler results in:', 'opts': ['Values 0-1', 'Mean=0, Std=1', 'All positive', 'Max=100'], 'ans': 1},
        {'q': 'MinMaxScaler results in:', 'opts': ['Mean=0', 'Values 0-1', 'Std=1', 'Unchanged'], 'ans': 1},
        {'q': 'One-Hot Encoding creates:', 'opts': ['Ordinal numbers', 'Binary columns', 'Continuous values', 'Missing values'], 'ans': 1},
    ],
    'algorithms': [
        {'q': 'Which does NOT need scaling?', 'opts': ['SVM', 'KNN', 'Decision Tree', 'Neural Network'], 'ans': 2},
        {'q': 'Random Forest is a ___ method:', 'opts': ['Boosting', 'Bagging', 'Stacking', 'Simple'], 'ans': 1},
        {'q': 'Support vectors are:', 'opts': ['All data points', 'Points on margin boundary', 'Outliers only', 'Centroids'], 'ans': 1},
        {'q': 'Naive Bayes assumes:', 'opts': ['Feature dependence', 'Feature independence', 'Linear boundary', 'No assumptions'], 'ans': 1},
    ],
}

# ============================================
# ROUTES
# ============================================

@app.route('/')
def index():
    """Home page with guided navigation"""
    return render_template('index.html',
                          topics=TOPICS,
                          formulas=FORMULAS[:6])

@app.route('/definitions')
def definitions():
    """All key definitions"""
    # Get podcast status for each definition category
    podcasts = {}
    for key in DEFINITIONS.keys():
        podcast_id = f'definitions-{key}'
        podcasts[podcast_id] = {
            'exists': audio_exists(podcast_id),
            'url': get_audio_path(podcast_id) if audio_exists(podcast_id) else None
        }
    return render_template('definitions.html',
                          definitions=DEFINITIONS,
                          podcasts=podcasts)

@app.route('/formulas')
def formulas():
    """All formulas with examples"""
    podcast_id = 'formulas-all'
    return render_template('formulas.html',
                          formulas=FORMULAS,
                          podcast_exists=audio_exists(podcast_id),
                          podcast_url=get_audio_path(podcast_id) if audio_exists(podcast_id) else None)

@app.route('/practice')
def practice():
    """Practice problems with calculations"""
    # Get podcast status for practice topics
    practice_podcast_ids = ['practice-confusion-matrix', 'practice-statistics',
                           'practice-scaling', 'practice-imbalance']
    podcasts = {}
    for pid in practice_podcast_ids:
        podcasts[pid] = {
            'exists': audio_exists(pid),
            'url': get_audio_path(pid) if audio_exists(pid) else None
        }
    return render_template('practice.html',
                          problems=PRACTICE_PROBLEMS,
                          quizzes=QUIZ_DATA,
                          podcasts=podcasts)

@app.route('/quick-reference')
def quick_reference():
    """Quick reference sheet"""
    podcast_id = 'quickref-all'
    return render_template('quick-reference.html',
                          formulas=FORMULAS,
                          definitions=DEFINITIONS,
                          podcast_exists=audio_exists(podcast_id),
                          podcast_url=get_audio_path(podcast_id) if audio_exists(podcast_id) else None)

@app.route('/quiz/<topic>')
def quiz(topic):
    """Quiz for specific topic"""
    questions = QUIZ_DATA.get(topic, [])
    return render_template('quiz.html',
                          topic=topic,
                          questions=questions)

# ============================================
# PODCAST ROUTES
# ============================================

@app.route('/podcast')
def podcast():
    """Podcast page with all topics"""
    topics = get_available_topics()
    # Add status for each topic
    for topic in topics:
        topic['has_audio'] = audio_exists(topic['id'])
        topic['audio_url'] = get_audio_path(topic['id']) if topic['has_audio'] else None
    return render_template('podcast.html', topics=topics, scripts=PODCAST_SCRIPTS)


@app.route('/api/podcast/generate/<topic_id>', methods=['POST'])
def generate_podcast(topic_id):
    """Generate podcast audio for a topic"""
    try:
        if topic_id not in PODCAST_SCRIPTS:
            return jsonify({'error': 'Unknown topic'}), 404

        # Generate audio
        path = generate_topic_audio(topic_id)
        return jsonify({
            'success': True,
            'topic': topic_id,
            'audio_url': get_audio_path(topic_id)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/podcast/status/<topic_id>')
def podcast_status(topic_id):
    """Check if podcast audio exists"""
    return jsonify({
        'topic': topic_id,
        'exists': audio_exists(topic_id),
        'audio_url': get_audio_path(topic_id) if audio_exists(topic_id) else None
    })


@app.route('/api/podcast/topics')
def podcast_topics():
    """Get all available podcast topics"""
    topics = get_available_topics()
    for topic in topics:
        topic['has_audio'] = audio_exists(topic['id'])
        topic['audio_url'] = get_audio_path(topic['id']) if topic['has_audio'] else None
    return jsonify(topics)


@app.errorhandler(404)
def page_not_found(e):
    return render_template('index.html', topics=TOPICS, formulas=FORMULAS[:6]), 404


if __name__ == '__main__':
    print("=" * 40)
    print("DACS ML Study Guide")
    print("http://localhost:8082")
    print("=" * 40)
    app.run(debug=True, host='0.0.0.0', port=8082)
