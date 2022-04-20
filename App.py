from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
    df = df.drop_duplicates(keep='first')
    # Features and Labels
    df['label'] = df['class'].map({'ham': 0, 'spam': 1})
    X = df['message']
    y = df['label']
    # Extract Feature With CountVectorizer
    tokenizer = RegexpTokenizer('\w+')
    sw = set(stopwords.words('english'))
    ps = PorterStemmer()
    def getStem(review):
        review = review.lower()
        tokens = tokenizer.tokenize(review) # breaking into small words
        removed_stopwords = [w for w in tokens if w not in sw]
        stemmed_words = [ps.stem(token) for token in removed_stopwords]
        clean_review = ' '.join(stemmed_words)
        return clean_review
    def getDoc(document):
        d = []
        for doc in document:
            d.append(getStem(doc))
        return d
    stemmed_doc = getDoc(X)
    cv = CountVectorizer()
    X = cv.fit_transform(X)  # Fit the Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # Naive Bayes Classifier
    modelnb = MultinomialNB()
    modelnb.fit(X_train, y_train)
    modelnb.score(X_test, y_test)
    modelrf = RandomForestClassifier(n_estimators=200, criterion='gini', min_samples_split=5, min_samples_leaf=2, max_features='auto', bootstrap=True, n_jobs=-1, random_state=42)
    modelrf.fit(X_train, y_train)
    modelrf.score(X_test, y_test)
    modeldt = DecisionTreeClassifier(random_state=100)
    modeldt.fit(X_train, y_train)
    modeldt.score(X_test, y_test)
    modellg = LogisticRegression()
    modellg.fit(X_train, y_train)
    modellg.score(X_test, y_test)
    modelada=AdaBoostClassifier()
    modelada.fit(X_train, y_train)
    modelada.score(X_test, y_test)
    vote = VotingClassifier(
    estimators = [("rf",modelrf),("log", modellg), ("Dt", modeldt), ("nG", modelnb),("ada",modelada)],
    voting='soft',
    weights=None,
    n_jobs=None,
    flatten_transform=True,
    )
    for clf in (modelrf, modellg, modeldt, modelnb, modelada, vote):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy_score(y_test, y_pred)
    
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
       
       
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('index.html', prediction= my_prediction)

if __name__ == '__main__':
    app.run(port=5001)