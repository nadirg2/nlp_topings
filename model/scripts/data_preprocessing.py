import re
from numpy import ndarray
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

from config import DataConfig, VectorizerConfig
from model.utils.logger import logging
from model.utils.pickler import save_object, load_object
from sklearn.feature_extraction.text import TfidfVectorizer


class DataPreprocessing:
    stop_words: set[str]
    lemmatizer: WordNetLemmatizer
    data_config: DataConfig

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.data_config = DataConfig

    def preprocess_text(self, text):
        # Удаление специальных символов и чисел
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Приведение к нижнему регистру
        text = text.lower()
        # Удаление стоп-слов и лемматизация
        tokens = text.split()
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)


    def preprocess_data(self):
        logging.info("Importing data for preprocessing")
        train_df = pd.read_csv(self.data_config.TRAIN_DF_RAW_PATH)
        test_df = pd.read_csv(self.data_config.TEST_DF_RAW_PATH)

        logging.info("Starting data preprocessing")

        df = pd.concat([train_df, test_df], ignore_index=True)
        
        df['text'] = df['Title'] + ' ' + df['Description']  # Объединяем заголовок и описание
        df = df.drop(['Title', 'Description'], axis=1)

        # Применяем предобработку к тексту
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        df = df.drop(['text'], axis=1)

        logging.info("Splitting dataframe into train and test")
        # Разделение данных на обучающую и тестовую выборки
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

        logging.info("Saving train dataframe")
        save_object(self.data_config.TRAIN_DF_PROCESSED_PATH, train_df)
        
        logging.info("Saving test dataframe")
        save_object(self.data_config.TEST_DF_PROCESSED_PATH, test_df)

        logging.info("Data preprocessing completed")


class Vectorizer:
    vectorizer: TfidfVectorizer
    config: VectorizerConfig

    def __init__(self):
        logging.info("Initializing vectorizer")
        self.vectorizer = TfidfVectorizer(max_features=5000)  # Ограничиваемся 5000 признаками для скорости
        self.config = VectorizerConfig


    def fit(self, df: pd.DataFrame | list[str]):
        logging.info("Fitting vectorizer")
        self.vectorizer.fit(df)


    def transform(self, df: pd.DataFrame | list[str] | ndarray):
        logging.info("Transforming by vectorizer")
        return self.vectorizer.transform(df)


    def fit_transform(self, df: pd.DataFrame | list[str] | ndarray):
        logging.info("Fitting vectorizer and transforming")
        return self.vectorizer.fit_transform(df)


    def save(self):
        logging.info("Saving vectorizer to file")
        save_object(self.config.VECTORIZER_PATH, self.vectorizer)


    def load(self):
        logging.info("Loading vectorizer from file")
        self.vectorizer = load_object(self.config.VECTORIZER_PATH)

