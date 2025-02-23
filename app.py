import os
from flask import Flask, request, render_template
from config import ModelConfig
from model.pipelines.predict_pipeline import PredictPipeline
from model.pipelines.train_pipeline import TrainPipeline
from model.utils.logger import logging


if not os.path.exists(ModelConfig.MODEL_PATH):
    logging.info("Model is not found. Start training...")
    train_pipeline = TrainPipeline()
    train_pipeline.train()  # Запуск обучения

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        logging.info("Processing new POST request...")
        text = request.form['text']
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(text)

        return render_template('index.html', prediction=prediction, text=text)

    logging.info("Processing new GET request...")
    return render_template('index.html', prediction=None, text=None)

if __name__ == '__main__':
    # train_pipeline = TrainPipeline()
    # train_pipeline.train()
    logging.info("Running flask app")
    app.run(debug=True)
