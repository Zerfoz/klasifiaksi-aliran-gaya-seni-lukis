import numpy as np
import tensorflow as tf

from flask import Flask, request, render_template
from PIL import  Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():

    return render_template('index.html')

@app.route('/lukisan', methods = ['POST'])

def batik_classifier():
    #Ambil gambar yang dikirim pas request
    image_request = request.files['image']

    #konversi gambar menjadi array
    image_pil = Image.open(image_request)

    #ngeresize gambar
    expected_size = (224,224)
    resized_image_pil = image_pil.resize(expected_size)

    #generate array dengan numpy
    image_array = np.array(resized_image_pil)
    rescaled_image_array = image_array/255.
    batched_rescaled_image_array = np.array([rescaled_image_array])
    #print(batched_rescaled_image_array.shape)

    #load model
    loaded_model = tf.keras.models.load_model("keras_model.h5")
    #print(loaded_model.get_config())
    result = loaded_model.predict(batched_rescaled_image_array)
    get_formated_predict_result(result)
   

    return get_formated_predict_result(result)

def get_formated_predict_result(predict_result) :
    class_indices = {'Vincent Van Gogh - Non-Impressionism': 0, 'Leonardo Da Vinci - High_Renaissance': 1, 'Edvard Munch - Expressionism': 2}
    inverted_class_indices = {}

    for key in class_indices:
        class_indices_key = key
        class_indices_value = class_indices[key]
        inverted_class_indices[class_indices_value] = class_indices_key

    processed_predict_result = predict_result[0]
    maxIndex = 0
    maxValue = 0

    for index in range(len(processed_predict_result)):
        if processed_predict_result[index] > maxValue:
            maxValue = processed_predict_result[index]
            maxIndex = index

    return inverted_class_indices[maxIndex]

if __name__ == "__main__":
    app.run()

