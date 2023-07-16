from flask import Flask, render_template, request,jsonify
import numpy as np
from keras.models import load_model
import cv2
import os
from sklearn.preprocessing import LabelEncoder
from process_seg import coff ,ImageProcessor2
import base64
from prep import ImageProcessor
import joblib
import json
#activate: .venv\Scripts\activate


# Load the classification model
#classification
def predict_class(path):
    # Get the uploaded image file
    model = load_model("models\classification\classification.h5")
    IMG_SIZE = 150

    # Read the image, resize it, and preprocess it for classification
    img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    Xtt = np.array(new_array).reshape(-1, IMG_SIZE, IMG_SIZE)
    Xtt = Xtt.reshape(-1, 150, 150, 1)

    # Make predictions using the classification model
    predict_x = model.predict(Xtt)
    classes_pred = np.argmax(predict_x, axis=1)

    # Map the predicted class index to the corresponding label
    classes = {0: "Glioma", 1: "Meningioma", 2: "No tumor", 3: "Pituitary"}
    class_index = classes_pred[0]
    class_label = classes.get(class_index, "Unknown")

    # Return the result to the 'service.html' template
    return (class_label)
# def grading(path,laterality,tumor_location,gender,age_at_initial_pathologic):
def grading(path,laterality,tumor_location,gender,age_at_initial_pathologic):
    my_dict = {
        'laterality': laterality,
        'tumor_location': tumor_location,
        'gender': gender,
        'age_at_initial_pathologic': int(age_at_initial_pathologic)
    }
    proc=ImageProcessor()
    model = load_model("models\grading\mdoel_2.h5")

    x =proc.process_img(path)
    test=proc.process_data(my_dict)
    # x.shape
    ##################################################################
    lbl=model.predict([x, test])
    lbl = np.argmax(lbl, axis = 1) 
    predlbl = 'low grade' if lbl == 0 else 'high grade' 
    return predlbl
def predict_protocol(age, previous_treatments, tumor_type, level, spread_of_tumor):
    tumor_type_encoder = LabelEncoder()
    level_encoder = LabelEncoder()

    tumor_type_mapping = {1: "Glioma", 2: "Pituitary", 3: "Meningioma"}
    level_mapping = {0: "low grade", 1: "high grade"}
    previous_treatments=int(previous_treatments)
    spread_of_tumor=int(spread_of_tumor)
    # Fit the encoders on the available categories
    tumor_type_encoder.fit(list(tumor_type_mapping.values()))
    level_encoder.fit(list(level_mapping.values()))

    # Encode the input variables
    encoded_tumor_type = tumor_type_encoder.transform([tumor_type])[0]
    encoded_level = level_encoder.transform([level])[0]

    model = joblib.load('models\protocol\model.pkl')
    data = [[age, previous_treatments, encoded_tumor_type, encoded_level, spread_of_tumor]]
    protocol = model.predict(data)[0]
    return protocol

#Segmentation 
def segmentation(path):
    losses = coff(smooth=1.0)
    proc=ImageProcessor2()
    model = load_model('models\segmintation\seg_class.h5', custom_objects={'bce_dice_loss': losses.bce_dice_loss, 'iou': losses.iou, 'dice_coef': losses.dice_coef})
    # Load the original image
    image = cv2.imread(path)

    # Print the shape of the original image
    # print('Original image shape:', image.shape)

    # Preprocess the image
    image_path = path
    image_ex = proc.preprocess_image(image_path)

    # Print the shape of the preprocessed image
    # print('Preprocessed image shape:', image_ex.shape)

    # Assume `model` is a Keras model that predicts a segmentation mask
    pred = model.predict(image_ex)

    # Print the shape of the predicted segmentation mask
    # print('Segmentation mask shape:', pred.shape)

    # Merge the segmentation mask with the original image
    merged = proc.merging(image, pred[0])

    return merged
app = Flask(__name__, template_folder='template')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/index2', methods=['GET'])
def index2():
    return render_template('index.html')
@app.route('/service', methods=['GET'])
def service():
    return render_template('service.html')

@app.route('/analysis', methods=['GET'])
def analysis():
    return render_template('analysis.html')

@app.route('/dashboard', methods=['GET'])
def dashboard():
    return render_template('dashboard.html')

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET'])
def contact():
    return render_template('contact.html')


# Classification route
@app.route('/predict', methods=['POST'])
def predict():
    # print("predict")
    # print(request)
    imgfile = request.files['file']
    
    laterality = request.form.get('laterality')
    tumor_location = request.form.get('tumor_location')
    gender = request.form.get('gender')
    age = request.form.get('age')

    Previous_Treatment=request.form.get('Previous_Treatment')
    Spread_of_tumor=request.form.get('Spread_of_tumor')
    # print(Spread_of_tumor)
    # Create the 'images' directory if it doesn't exist
    os.makedirs('./images/', exist_ok=True)

    # Set the file path to save the uploaded image
    path = "./images/" + imgfile.filename

    # Save the image file if it doesn't already exist
    if not os.path.exists(path):
        imgfile.save(path)
    class_label=predict_class(path)
    if class_label=="No tumor":
        output = {
        'class_label': class_label,
        
    }
    else:
        seg_img=segmentation(path)
        seg_img = base64.b64encode(cv2.imencode('.png', seg_img)[1]).decode()
        grade=grading(path,laterality,tumor_location,gender,age)
        protocol=predict_protocol(age,Previous_Treatment,class_label,grade,Spread_of_tumor)
        
        output = {
            'class_label': class_label,
            'grade': grade,
            'protocol': protocol,
            'image':seg_img
        }
    
    
    # return render_template('service.html', classes_pred=class_label,grade=grade,protocol=protocol)
    return jsonify(output)


if __name__ == '__main__':
    app.run(debug=True, port=3000)
