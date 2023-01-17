from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from neural_network import Net
import torch
from utils import get_probability, plot_model_prediction, generate_feature_maps

UPLOAD_FOLDER = 'static/imgs/uploads'

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 16 * 800 * 800
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def check_file_extension(filename):
    allowed_file_extensions = ('jpg', 'jpeg', 'png', 'jfif')
    if filename.lower().endswith(allowed_file_extensions):
        return "."
    else:
        return


def load_model():

    model = Net()
    model.load_state_dict(torch.load('models/trained_model_epoch_5.pth', map_location=torch.device('cpu') ))
    model.eval()

    return model


@app.route('/', methods=['GET', 'POST'])
def landing():
    return render_template('landing.html')

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')


@app.route('/model', methods=['GET', 'POST'])
def model():

    return render_template('model.html')

@app.route('/improvements', methods=['GET', 'POST'])
def improvements():

    return render_template('improvements.html')

@app.route('/uploads', methods=['GET', 'POST'])
def uploads():

    return render_template('upload.html')

@app.route('/uploads_pressed', methods=['GET', 'POST'])
def uploads_pressed():

    if request.files['file'].filename == "":
        return render_template('upload.html', error="Please select a file")

    else:

            file = request.files['file']
            filename = secure_filename(file.filename)

            if file and check_file_extension(file.filename):

                model = load_model()

                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))


                prediction = get_probability(model, filename)
                plot_model_prediction(prediction, filename)
                generate_feature_maps(model, filename)

                return render_template('prediction.html', uploaded_img = filename, model_prediction_name = filename.split(".")[0], feature_maps = filename.split(".")[0])

            else:
                return render_template('upload.html', error="Please upload an image with the type 'jpg', 'jpeg', 'png' or 'jfif'")


@app.route('/thesis', methods=['GET', 'POST'])
def thesis():
    return render_template('thesis.html')

@app.route('/advent', methods=['GET', 'POST'])
def advent():
    return render_template('advent.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    return render_template('contact.html')

@app.route('/model_fail', methods=['GET', 'POST'])
def model_fail():
    return render_template('model_fail.html')

if __name__ == '__main__':
    app.run(debug = True)