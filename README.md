# NBA Game Predictor

This project predicts NBA game outcomes using a neural network based on team statistics. It includes a model training script, a Flask web application, and a simple front-end interface.

## Features

- Neural network model for predicting game outcomes
- Web interface for selecting teams and displaying predictions
- Display of classification report metrics

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/nba-game-predictor.git
    cd nba-game-predictor
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download dataset**:
    - Ensure `nba_games.csv` is in the project root directory.

## Usage

### Model Training

1. **Run the training script**:
    ```bash
    python main_script.py
    ```

### Web Application

1. **Run the web application**:
    ```bash
    python app.py
    ```

2. **Open** `http://127.0.0.1:5000/` in a web browser.

3. **Select teams** and click "Predict" to see the game outcome prediction.

## File Descriptions

- `main_script.py`: Trains the neural network model.
- `app.py`: Flask web application.
- `templates/index.html`: Web application interface.
- `static/style.css`: CSS for the web application.
- `nba_games.csv`: Historical NBA game statistics.
- `neural_network_model.h5`: Trained model.
- `scaler.pkl`: Data normalization scaler.
- `classification_report.pkl`: Model evaluation report.
