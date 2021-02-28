from flask import Flask,render_template,request
import pickle
import numpy as np


model = pickle.load(open('prediction.pkl','rb'))

app = Flask(__name__)



@app.route('/')
def home():
    return render_template('campaignform.html')


@app.route('/result', methods=['GET','POST'])
def get_prediction():
    data1=request.form['name']
    data2=request.form['main_category']
    data3=request.form['duration']
    data4=request.form['goal']
    data5=request.form['purpose']
    arr=np.array([[data1,data2,data3,data4,data5]])
    pred = model.predict(arr)
    return render_template('result.html',data=pred)


if __name__ == "__main__":
    app.run(debug=True)