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
        result = airline_func(res_airlineName)  
        return render_template('finalResult.html', result=result)



if __name__ == '__main__':
    app.run(debug=True)