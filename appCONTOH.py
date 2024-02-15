from flask import Flask, render_template
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

def generate_plot(data):
    # Generate a sample plot

    hasil_kalkulasi = sum(data) 

    x= [1,2,3,4,5]
    y= [10,20,30,35,50]

    plt.plot(x,y)
    plt.xlabel(u"X-Axis Label")
    plt.ylabel("Y-Axis Label")
    plt.title("My Plot Title")

    # simpan plot ke dalam buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)  # rewind the buffer to start of content

    # encode plot ke dalam base64
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png).decode()
    buffer.close()

    return graph, hasil_kalkulasi


@app.route('/')
def index():
    data = [10,20,30,40,50]
    plot, hasil_kalkulasi = generate_plot(data)
    return render_template('indexCONTOH.html',  plot=plot, 
                           hasil_kalkulasi = hasil_kalkulasi)


if __name__ == '__main__':
    app.run(debug=True)