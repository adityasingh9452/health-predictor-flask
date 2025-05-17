from flask import Flask, render_template_string, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Target-feature mapping
targets_features = {
    'prediabetic': ['Glucose', 'BloodPressure', 'Insulin'],
    'diabetes': ['Glucose', 'Insulin'],
    'prehypertension': ['BloodPressure', 'Insulin'],
    'hypertension': ['BloodPressure', 'Glucose']
}

# Models to use
model_names = ['LogisticRegression', 'RandomForest', 'SVM']

# Get all unique input features
all_features = sorted(set(f for features in targets_features.values() for f in features))

# HTML templates as strings
form_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Health Risk Prediction</title>
</head>
<body>
    <h2>Enter Your Info</h2>
    <form method="post">
        <label>Name:</label><br>
        <input type="text" name="name" required><br><br>

        <label>Age:</label><br>
        <input type="number" name="age" required><br><br>

        {% for feature in features %}
            <label>{{ feature }}:</label><br>
            <input type="number" step="any" name="{{ feature }}" required><br><br>
        {% endfor %}

        <input type="submit" value="Predict">
    </form>
</body>
</html>
"""

result_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Prediction Result</title>
</head>
<body>
    <h2>Prediction Results for {{ name }}, Age {{ age }}</h2>

    {% for result in predictions %}
        <h3>{{ result.target }}</h3>
        <ul>
            {% for model, prediction in result.items() if model != 'target' %}
                <li>{{ model }}: {{ 'Yes (1)' if prediction == 1 else 'No (0)' }}</li>
            {% endfor %}
        </ul>
    {% endfor %}

    <br><a href="/">Try another</a>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get name and age (not used in prediction)
        name = request.form['name']
        age = request.form['age']

        # Get feature inputs
        feature_data = {}
        for feature in all_features:
            try:
                feature_data[feature] = float(request.form[feature])
            except ValueError:
                return f"Invalid input for {feature}. Please enter a number."

        X_user = pd.DataFrame([feature_data])
        predictions = []

        for target, features in targets_features.items():
            target_preds = {'target': target}
            for model_name in model_names:
                model_file = f'{target}_{model_name}.joblib'
                if not os.path.exists(model_file):
                    target_preds[model_name] = 'Model not found'
                    continue
                model = joblib.load(model_file)
                pred = model.predict(X_user[features])[0]
                target_preds[model_name] = int(pred)
            predictions.append(target_preds)

        return render_template_string(result_html, name=name, age=age, predictions=predictions)

    return render_template_string(form_html, features=all_features)

if __name__ == '__main__':
    app.run(debug=True)