import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from warnings import simplefilter

from flask import Flask, request, jsonify

# Ignore user warning
simplefilter(action='ignore', category=UserWarning)

def gradient_boosting(hospitalized_now: int):

    df = pd.read_csv("clean_covid_data.csv", index_col=[0])
    df = df.dropna()

    x = df[["hospitalizedCurrently"]]
    y = df["deathIncrease"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, shuffle=True)
    model = GradientBoostingRegressor(n_estimators=250, learning_rate=0.01, min_samples_leaf=20)

    model.fit(x_train, y_train)

    return round(model.predict([[hospitalized_now]])[0], 2)


app = Flask(__name__)

@app.route('/predict/<hospitalized_now>', methods=['GET'])
def predict(hospitalized_now):
    
        if request.method == 'GET':
    
            hospitalized_now = int(hospitalized_now)
            death_increase = gradient_boosting(hospitalized_now)
    
            return jsonify({
                "death_increase": death_increase
            })
        
if __name__ == '__main__':
    app.run(debug=True)

    
