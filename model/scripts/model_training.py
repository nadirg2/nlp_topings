from sklearn.linear_model import LogisticRegression
from config import DataConfig, ModelConfig
from model.scripts.data_preprocessing import Vectorizer
from model.utils.logger import logging
from model.utils.pickler import save_object, load_object


class TrainModel:
    data_config: DataConfig
    model_config: ModelConfig


    def __init__(self):
        self.data_config = DataConfig
        self.model_config = ModelConfig


    def train(self):
        logging.info("Importing data for training")
        train_df = load_object(self.data_config.TRAIN_DF_PROCESSED_PATH)

        X_train = train_df['processed_text']
        y_train = train_df['Class Index']

        # Векторизация текста (TF-IDF)
        vectorizer = Vectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)

        # Обучение модели
        logging.info("Initializing model")
        model = LogisticRegression(max_iter=1000)  # Увеличиваем max_iter для сходимости

        logging.info("Fitting model")
        model.fit(X_train_vec, y_train)

        logging.info("Saving model")
        save_object(self.model_config.MODEL_PATH, model)

        vectorizer.save()

        print("Model training completed!")
