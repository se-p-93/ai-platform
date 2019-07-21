import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from nltk import word_tokenize
from nltk.corpus import stopwords

stop = set(stopwords.words('spanish'))

class ProfessionClassifier:

    def __init__(self):
        self.train = None
        self.test = None

    def import_dataset(self):
        prof = open("data\Csv_Profesiones_02.csv", encoding="utf8")

        ls = prof.readline().lower()

        lines_without_anotation = []
        lines_with_anotation = []
        lines_with_error = []

        while ls:
            text = ls.replace("\n", "")
            tx = text.lower().split("||")
            if len(tx) <= 2:
                lines_without_anotation.append(tx)
            elif len(tx) == 4:
                tx[2] = tx[2].replace('|', '')
                tx[2] = tx[2].replace(' ', '')
                tx[3] = tx[3].replace('|', '')
                tx[3] = tx[3].replace(' ', '')
                lines_with_anotation.append(tx)
            else:
                lines_with_error.append(tx)
            ls = prof.readline().lower()

        df_with = pd.DataFrame(lines_with_anotation, columns=('id', 'text', 'category_1', 'category_2'))
        df = pd.DataFrame(df_with, columns=('id', 'text', 'category_1', 'category_2'))

        df.category_1.unique()
        others = self.reduce_labels_profesion(0.0015, df)
        df = self.change_labels_profesion(df, others)
        self.train, self.test = train_test_split(df, test_size=0.15, random_state=29)

    def reduce_labels_profesion(self, fraction, df):
        others = []
        min_len = len(df['category_1']) * fraction
        for key in df['category_1'].value_counts().keys():

            if df['category_1'].value_counts()[key] < min_len:
                others.append(key)
            else:
                pass
        return others

    def change_labels_profesion(self, df, others):
        for i, row in df.iterrows():
            if row['category_1'] in others:
                df.loc[i, 'category_1'] = 'others'

        return df

    def label_encoding(self, df):
        lb = LabelEncoder()
        df_enconded = lb.fit_transform(df)
        return df_enconded

    def training(self):
        self.import_dataset()
        y_train = self.label_encoding(self.train['category_1'].tolist())
        pipeline = Pipeline([
            ('selector', ItemSelector(key='text')),
            ('StopWords', Stopwords()),
            ('vect', CountVectorizer(max_features=None, ngram_range=(1, 2))),
            ('tfidf', TfidfTransformer()),
            # Use a SVC classifier on the combined features
            ('clf', CalibratedClassifierCV(LinearSVC(random_state=42, multi_class='ovr'))),
        ])
        pipeline.fit(self.train, y_train)
        y_score = pipeline.predict_proba(self.test)
        y_test = self.label_encoding(self.test['category_1'].tolist())
        f1 = f1_score(y_test, [np.argmax(i) for i in y_score], average="micro")
        print(f1)
        joblib.dump(pipeline, 'model_profession.pkl')

        return f1


class Stopwords(BaseEstimator, TransformerMixin):
    """Remove stopwords from each text"""

    def fit(self, x, y=None):
        return self

    def transform(self, texts):
        result = []
        stop = set(stopwords.words('spanish'))
        for text in texts:
            text_stopwords_removed = [i for i in word_tokenize(text.lower()) if i not in stop]
            result.append(' '.join(text_stopwords_removed))
        return result


class ItemSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


if __name__ == '__main__':
    with mlflow.start_run():
        classifier = ProfessionClassifier()
        f1 = classifier.training()
        mlflow.log_metric("F1 Score", f1)
