import os
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from keras._tf_keras.keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import InputRequired




app = Flask(__name__)
app.config['SECRET_KEY'] = 'secretiveKey'
app.config['UPLOAD_FOLDER'] = 'uploads'

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Predict")



model = load_model('BrainTumor10Epochs.h5')
print('Model loaded. Check http://127.0.0.1:5000')

def get_className(classNo):
    if classNo == 0:
        return "Patient Does Not  Have Brain Tumor"
    elif classNo == 1:
        return "Patient Has Brain Tumor"
    

def getResult(img):
    image = cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result = (model.predict(input_img)).astype("int32")
    return result


@app.route('/', methods=['GET', "POST"])
def index():
    form = UploadFileForm()
    return render_template('index.html', form=form)


@app.route('/predict', methods=['GET', "POST"])
def upload():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = request.files['file'] # first grab the file
        file_path = os.path.join(os.path.dirname(__file__), app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(file_path) # then save file

        value = getResult(file_path)
        result = get_className(value)
        return render_template('index.html', res=result, form=form, image=file.filename)
    

@app.route('/uploads/<filename>')
def send_uploaded_file(filename=''):
    from flask import send_from_directory
    image = "uploads/"+ filename
    image = Image.open(image)
    image = image.resize((200, 250))
    image.save("uploads/"+filename, format="JPEG")
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)









if __name__ == '__main__':
    app.run(debug=True)