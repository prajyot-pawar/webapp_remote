import sklearn
import pickle
import flask
from flask import Flask
from flask import Flask, request, render_template
from flask import request, jsonify


app = Flask(__name__)

cv = pickle.load(open('mycv.pkb', 'rb'))

predictions = {
    0: 'Business News',
    1: 'Tech News',
    2: 'Politics News',
    3: 'Sports News',
    4: 'Entertainment News'
}


@app.route('/', methods=['GET', 'POST'])
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    news_data = request.form['news']
    # result = "your news is  :: " + news_data
    temp = cv.transform([news_data])
    loaded_model = pickle.load(open('model.pkl', 'rb'))
    loaded_ensemble_model = pickle.load(open('ensemble_model.pkl', 'rb'))
    result = loaded_model.predict(temp)
    result_1 = loaded_ensemble_model.predict(temp)
    if result == 0:
        result = result_1 = "Business News"
    elif result == 1:
        result = result_1 = "Tech News"
    elif result == 2:
        result = result_1 = "Politics News"
    elif result == 3:
        result = result_1 = "Sports News"
    elif result == 4:
        result = result_1 = 'Entertainment News'
    return render_template("predict.html", result=result, result_1=result_1)


if __name__ == '__main__':
    app.run(debug=True)
