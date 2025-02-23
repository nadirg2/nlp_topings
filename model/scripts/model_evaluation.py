from sklearn.metrics import classification_report, accuracy_score

from config import DataConfig, ModelConfig
from model.scripts.data_preprocessing import Vectorizer
from model.utils.logger import logging
from model.utils.pickler import load_object

class ModelEvaluation:
    data_config: DataConfig
    model_config: ModelConfig


    def __init__(self):
        self.data_config = DataConfig
        self.model_config = ModelConfig


    def evaluate_model(self):
        logging.info("Importing data for evaluation")

        test_df = load_object(self.data_config.TEST_DF_PROCESSED_PATH)
        model = load_object(self.model_config.MODEL_PATH)
        vectorizer = Vectorizer()
        vectorizer.load()

        logging.info("Predicting test data")
        # Предсказание и оценка
        X_test = test_df['processed_text']
        y_test = test_df['Class Index']

        X_test_vec = vectorizer.transform(X_test)

        y_pred = model.predict(X_test_vec)

        logging.info("Scoring metrics")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        print("\nClassification Report:")
        report = classification_report(y_test, y_pred)
        logging.info('\nClassification Report:\n' + report)
        print(report)

        logging.info("Model evaluating completed")
