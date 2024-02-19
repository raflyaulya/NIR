from flask import Flask, render_template, request, url_for 
from file_analyze import airline_func           # change ur analysis library

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/finalResult', methods=['POST'])
def finalResult():
    if request.method == 'POST':
        res_airlineName = request.form['airlineName']
        plots, result_positive, result_negative = airline_func(res_airlineName)  
        return render_template('finalResult.html',plots=plots, 
                               result_positive=result_positive, result_negative = result_negative)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
    # d:/ИПМКН/Семестр 5/НИР/NIR/app.py