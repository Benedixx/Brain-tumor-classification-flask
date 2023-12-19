import os
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as tf_image
from flask import Flask, request, jsonify, render_template_string, render_template


app = Flask(__name__)

app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])
app.config['MODEL_BRAIN_TUMOR'] = '\\models\\model_brain_tumor.h5'
app.config['UPLOAD_FOLDER'] = '\\static\\uploads'

model_brain_tumor = load_model(app.config['MODEL_BRAIN_TUMOR'], compile=False)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']
           
@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'Data': {
            'Project': 'Brain Tumor Classification',
            'Owner' : 'Benedixx'
        }
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == 'POST':
        reqImage = request.files['image']
        if reqImage and allowed_file(reqImage.filename):
            filename = secure_filename(reqImage.filename)
            reqImage.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img = Image.open(image_path).convert('RGB')
            img = img.resize((256, 256))
            x = tf_image.img_to_array(img)
            x /= 255
            x = np.expand_dims(x, axis=0)
            classification = model_brain_tumor.predict(x)
            class_list = ['glioma','meningioma','no_tumor','pituitary']
            classification_class = class_list[np.argmax(classification[0])]
            os.remove(image_path)
            return jsonify({
                'status': {
                    'code': 200,
                    'message': 'Success predicting',
                    'data': { 'class': classification_class }
                }
            }), 200
        else :
            return jsonify({
                'status': {
                    'code': 400,
                    'message': 'Invalid file format. Please upload a JPG, JPEG, or PNG image.'
                }
            }), 400
    else:
        return jsonify({
            'status': {
                'code': 405,
                'message': 'Method not allowed'
            }
        }), 405
            
if __name__ == "__main__":
    app.run(debug=True)