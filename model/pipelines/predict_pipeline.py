from sklearn.linear_model import LogisticRegression
from config import ModelConfig, VectorizerConfig
from model.scripts.data_preprocessing import DataPreprocessing, Vectorizer
from model.scripts.model_evaluation import ModelEvaluation
from model.scripts.model_training import TrainModel
from model.utils.pickler import load_object

class PredictPipeline:
    CLASS_NUM_TO_TITLE = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}
    preprocessor: DataPreprocessing
    model_config: ModelConfig
    vectorizer_config: VectorizerConfig
    model: LogisticRegression
    vectorizer: Vectorizer


    def __init__(self):
        self.preprocessor = DataPreprocessing()
        self.model_config = ModelConfig()
        self.vectorizer_config = VectorizerConfig()

        self.model = load_object(self.model_config.MODEL_PATH)
        self.vectorizer = Vectorizer()
        self.vectorizer.load()


    def predict(self, text: str):
        processed_text = self.preprocessor.preprocess_text(text)
        X_vec = self.vectorizer.transform([processed_text])

        predicted_label_num = self.model.predict(X_vec)[0]

        return self.CLASS_NUM_TO_TITLE[predicted_label_num]
