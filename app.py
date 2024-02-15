from flask import Flask, render_template, request, url_for 
from file_analyze import airline_func           # change ur analysis library
import io
import base64

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/finalResult', methods=['POST'])
def finalResult():
    if request.method == 'POST':
        res_airlineName = request.form['airlineName']
        result, plot = airline_func(res_airlineName)  
        return render_template('finalResult.html', result=result,
                               plot=plot)


if __name__ == '__main__':
    app.run(debug=True)
    # d:/ИПМКН/Семестр 5/НИР/NIR/app.py