# from flask import Flask
# app = Flask(__name__)


# @app.route('/')

# def hello_world():
#     return 'tesst 123S'

def coba(test):

    # test = 'bob'
    lis = []

    if test == 'aka':
        lis.append('true')
    elif test == 'bob':
        lis.append('WRONG')
    else:
        lis.append('I D K')


    return lis[0]

test= input('enter coba: ')

print(coba(test))