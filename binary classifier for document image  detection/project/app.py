# app.py
from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
from predict import load_model, predict_document

UPLOAD_FOLDER = 'static/uploads'  
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('resnet18_model.pth')  


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)  

        
        prediction, confidence = predict_document(file_path, model)

       
        return jsonify({'prediction': prediction, 'confidence': confidence, 'image': file_path})

    return jsonify({'error': 'Invalid file type'})

if __name__ == "__main__":
    app.run(debug=True)
