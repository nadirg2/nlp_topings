# News Topic Prediction

This project is about predicting the topic of news articles using the AG News dataset. The prediction model is built using Logistic Regression and is deployed as a Flask web application.

## Project Structure

This structure is modular and organized, making it easy to maintain and scale the project. Each folder and file has a clear purpose, ensuring a clean and efficient workflow.

```
project/
│
├── logs/                  # Logs for tracking execution (e.g., training logs)
│
├── model/                 # Main folder for model-related files
│   ├── data/              # Data folder
│   │   ├── processed/     # Processed data (e.g., after preprocessing)
│   │   └── raw/           # Raw data (e.g., original datasets)
│   │
│   ├── notebooks/         # Jupyter Notebooks for analysis and experiments
│   │   └── toping_of_news.ipynb
│   │
│   ├── pipelines/         # Folder for data processing and prediction pipelines
│   │   ├── predict_pipeline.py  # Pipeline for making predictions
│   │   └── train_pipeline.py    # Pipeline for training the model
│   │
│   ├── saved_models/      # Folder for saved model files
│   │
│   ├── scripts/           # Scripts for data processing and model training
│   │   ├── __init__.py    # Package initialization file
│   │   ├── data_preprocessing.py  # Script for data preprocessing
│   │   ├── model_evaluation.py    # Script for model evaluation
│   │   └── model_training.py      # Script for model training
│   │
│   ├── utils/             # Utility functions and helpers
│   │   ├── __init__.py    # Package initialization file
│   │   ├── logger.py      # Utility for logging
│   │   └── pickler.py     # Utility for serialization/deserialization (e.g., pickle)
│   │
│   ├── templates/         # HTML templates for the web interface
│   │   └── index.html     # Main HTML template for the web interface
│   │
│   ├── .gitignore         # File to exclude specific files/folders from Git
│   ├── app.py             # Main Flask application file
│   ├── config.py          # Project configuration (paths, parameters)
│   ├── README.md          # Project documentation
│   └── requirements.txt   # Project dependencies
```

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/nadirg2/nlp_topings.git
    ```
2. Navigate to the project directory:
    ```sh
    cd nlp_topings
    ```
3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Run the Flask application:
    ```sh
    python run.py
    ```
2. Open your web browser and go to `http://127.0.0.1:5000`.

## Dataset

The [AG News dataset](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset) is used for training and testing the model. It contains news articles categorized into four classes: World, Sports, Business, and Sci/Tech.

## Model

The Logistic Regression model and TF-IDF vectorizer is used for predicting the topic of news articles. The model is trained on the AG News dataset and saved as `model.pkl`. Also implemented text preprocessing with using lemmatization, stop-word removal, and tokenization.

## License

This project is licensed under the MIT License.