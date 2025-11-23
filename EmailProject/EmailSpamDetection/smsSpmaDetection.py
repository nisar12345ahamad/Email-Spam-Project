import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import string
import nltk
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('stopwords')


df = pd.read_csv("spam (1).csv", encoding="latin1")
print(df.head(5))

# Data Cleaning
print(df.info())
print(df.columns.tolist())

df.columns = df.columns.str.strip()
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
print(df.sample(5))

# renaming the column
df.rename(columns={'v1':'target','v2':'text'}, inplace=True)
print(df.sample(5))

# Label Encoding
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])

# Missing value check
print(df.isnull().sum())

# Duplicate check + remove
df = df.drop_duplicates(keep='first')
print(df.duplicated().sum())

print(df['target'].value_counts())

import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(), labels=['ham', 'spam'], autopct="%0.2f")
plt.show()

# Download required nltk resources
print(nltk.download('punkt'))
print(nltk.download('stopwords'))


# Numeric characters
df['num_character'] = df['text'].apply(len)

# num of words
df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(str(x))))

# num of sentences
df['num_sentences'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(str(x))))

print(df[['num_character','num_words','num_sentences']].describe())

# Visualization
import seaborn as sns
sns.histplot(df[df['target'] == 0]['num_character'])
sns.histplot(df[df['target'] == 1]['num_character'], color='r')
plt.show()

sns.histplot(df[df['target'] == 0]['num_words'])
sns.histplot(df[df['target'] == 1]['num_words'], color='r')
plt.show()

sns.histplot(df[df['target'] == 0]['num_sentences'])
sns.histplot(df[df['target'] == 1]['num_sentences'], color='r')
plt.show()

sns.pairplot(df, hue='target')
plt.show()

# -------------------------------
#     DATA PRE-PROCESSING
# -------------------------------

ps = PorterStemmer()

def transform_text(text):

    # 1 → Ensure string
    if not isinstance(text, str):
        text = str(text)

    # 2 → Lowercase
    text = text.lower()

    # 3 → Tokenization
    text = nltk.word_tokenize(text)

    # 4 → Remove special chars
    y = []
    for i in text:
        if i.isalnum():        # keep only alphanumeric
            y.append(i)

    # 5 → Remove stopwords + punctuation
    y2 = []
    for i in y:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y2.append(i)

    # 6 → Stemming
    y3 = []
    for i in y2:
        y3.append(ps.stem(i))

    return " ".join(y3)


# APPLY TRANSFORM FUNCTION
df['transformed_text'] = df['text'].apply(transform_text)

print(df['transformed_text'])
print(df['transformed_text'].sample(5))
print(df.head())

from wordcloud import WordCloud
wc = WordCloud(width=500 , height = 500 , min_font_size = 10 , background_color='white')
spam_wc = wc.generate(df[df['target']==1]['transformed_text'].str.cat(sep=" "))
plt.figure(figsize=(15,6))
plt.imshow(spam_wc)
plt.axis("off")
plt.show()

wc = WordCloud(width=500 , height = 500 , min_font_size = 10 , background_color='white')
spam_wc = wc.generate(df[df['target']==0]['transformed_text'].str.cat(sep=" "))
plt.figure(figsize=(15,6))
plt.imshow(spam_wc)
plt.axis("off")
plt.show()

# collect all the spam word nikalne ke liye ...

spam_corpus = []
for msg in df[df['target']==1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)

print(len(spam_corpus))

from collections import Counter
top30 = Counter(spam_corpus).most_common(30)
print(top30)

spam_df = pd.DataFrame(top30 , columns=["word","count"])
plt.figure(figsize=(15,10))
sns.barplot(data=spam_df , x="count",y="word")
plt.title("Top 30 words in spam corpus")
plt.xlabel("Count")
plt.ylabel("Word")
plt.tight_layout()
plt.show()

ham_corpus = []
for msg in df[df['target']==0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)

print(len(ham_corpus))
top30 = Counter(ham_corpus).most_common(30)
print(top30)

spam_df1 = pd.DataFrame(top30 , columns=["word","count"])
plt.figure(figsize=(15,10))
sns.barplot(data=spam_df1 , x="count",y="word")
plt.title("Top 30 words in spam corpus")
plt.xlabel("Count")
plt.ylabel("Word")
plt.tight_layout()
plt.show()


# Model Building Model

from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['transformed_text'])

# X = cv.fit_transform(df['transformed_text'])
print(X.shape)

y = df['target'].values
print(y)

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , y ,test_size=0.2 , random_state=2)

from sklearn.naive_bayes import GaussianNB ,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score , confusion_matrix, precision_score
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

gnb.fit(X_train.toarray(),y_train)
y_pred1 = gnb.predict(X_test.toarray())
print(accuracy_score(y_test , y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test , y_pred1))

mnb.fit(X_train,y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test , y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test , y_pred2))

bnb.fit(X_train,y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test , y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test , y_pred3))

# Model of machine learning ...
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

svc = SVC(kernel='sigmoid',gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
rfc = RandomForestClassifier(n_estimators=50 , random_state=2)
abc = AdaBoostClassifier(n_estimators=50 , random_state=2)
lrc = LogisticRegression(solver='liblinear',penalty='l1')
bc = BaggingClassifier(n_estimators=50 , random_state=2)
etc = ExtraTreesClassifier(n_estimators=50 , random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50 , random_state=2)
xgb = XGBClassifier(random_state =2)

# Made dictionary .
clfs = {
    'SVC' : svc ,
    'KN' : knc,
    'NB' :mnb,
    'DT' : dtc ,
    'LR' : lrc,
    'RF' : rfc,
    'AdaBoost': abc ,
    'BgC' : bc,
    'ETC' : etc,
    'GBDT': gbdt ,
    'xgb' : xgb
}

def train_classifier(clf , X_train , y_train , X_test , y_test):
    clf.fit(X_train , y_train)
    y_pred = clf.predict(X_test)

    accuracy1 = accuracy_score(y_test , y_pred)
    precision = precision_score(y_test , y_pred , average='binary',pos_label=1)
    return accuracy1 , precision

print(train_classifier(svc , X_train, y_train , X_test , y_test))

accuracy_list = []
precision_list = []

for name , clf in clfs.items():
    current_accuracy , current_precision = train_classifier(clf,X_train,y_train,X_test,y_test)

    print("for :" , name)
    print("Accuracy1:",current_accuracy)
    print("precision :",current_precision)

    accuracy_list.append(current_accuracy)
    precision_list.append(current_precision)
    
performance_df = pd.DataFrame({
    'Algorithms':list(clfs.keys()),
    'Accuracy1':accuracy_list,
    'precision':precision_list 
    }).sort_values('Accuracy1',ascending=False)
print(performance_df)


# Melt for catplot
performance_melted = performance_df.melt(id_vars='Algorithms', 
                                         value_vars=['Accuracy1', 'precision'])
sns.catplot(
    x='Algorithms',
    y='value',
    hue='variable',
    data=performance_melted,
    kind='bar',
    height=5
)
plt.ylim(0.5,1.0)
plt.xticks(rotation = "vertical")
plt.show()

import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))