from flask import Flask, request, render_template
from model.pipelines.predict_pipeline import PredictPipeline
from model.pipelines.train_pipeline import TrainPipeline

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Получение текста из формы
        text = request.form['text']

        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(text)

        # Возвращаем результат
        return render_template('index.html', prediction=prediction, text=text)

    return render_template('index.html', prediction=None, text=None)

if __name__ == '__main__':
    # train_pipeline = TrainPipeline()
    # train_pipeline.train()

    app.run(debug=True)
