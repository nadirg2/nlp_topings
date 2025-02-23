from model.scripts.data_preprocessing import DataPreprocessing
from model.scripts.model_evaluation import ModelEvaluation
from model.scripts.model_training import TrainModel


class TrainPipeline:
    def __init__(self):
        pass

    def train(self):
        preprocessor = DataPreprocessing()
        preprocessor.preprocess_data()

        train_model = TrainModel()
        train_model.train()

        model_evaluation = ModelEvaluation()
        model_evaluation.evaluate_model()

