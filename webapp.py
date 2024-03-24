from flask import Flask, render_template, request, url_for 
from urllib.parse import quote
from urllib.parse import quote as url_quote
from file_analyze import airline_func
from file_analyze_Negative import airline_func_Negative
from file_analyze_Positve import airline_func_Positive
import io
import base64


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/rules', methods=['POST'])
def rules():
    if request.method == 'POST':
        return render_template('rules.html')
    

@app.route('/choice', methods=['POST'])
def choice():
    if request.method == 'POST':
        return render_template('choice.html')

# @app.route('/finalResult_negative', methods=['POST'])
# def finalResult_negative():
#     if request.method == 'POST':
#         res_airlineName = request.form['airlineName']
#         plots, result_negative= airline_func_Negative(res_airlineName)  
#         return render_template('finalResult_negative.html',plots=plots, result_negative = result_negative)
    

# @app.route('/finalResult_Positive', methods=['POST'])
# def finalResult_Positive():
#     if request.method == 'POST':
#         res_airlineName = request.form['airlineName']
#         plots, result_positive= airline_func_Positive(res_airlineName)  
#         return render_template('finalResult_Positive.html',plots=plots, result_positive = result_positive)


# =====================================================================
    
@app.route('/finalResult', methods=['POST'])
def finalResult():
    if request.method == 'POST':
        res_airlineName = request.form['airlineName']         
        selectSentiment = request.form['selectSentiment']
        plots , res_sentiment = airline_func(res_airlineName, selectSentiment)
        return render_template('finalResult.html', res_airlineName= res_airlineName, selectSentiment= selectSentiment,
                                plots=plots, res_sentiment=res_sentiment)
    

                            #    res_airlineName=res_airlineName
        # res_airlineName, selectSentiment =  request.form[['airlineName', 'selectSentiment']]
        # plots, result_positive, result_negative= airline_func(res_airlineName, selectSentiment)  
        # return render_template('finalResult.html',plots=plots, 
        #                        result_positive=result_positive, result_negative = result_negative)







if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0') #debug = TRUE
    # d:/ИПМКН/Семестр 5/НИР/NIR/app.py