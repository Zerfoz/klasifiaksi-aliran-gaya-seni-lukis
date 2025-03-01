import os
import numpy as np
import tensorflow as tf

from flask import Flask, request,  render_template, jsonify
from PIL import  Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/lukisan', methods = ['POST'])

def batik_classifier():
    #Ambil gambar yang dikirim pas request
    image_request = request.files['image']

    #konversi gambar menjadi array
    image_pil = Image.open(image_request)

    #ngeresize gambar
    expected_size = (256,256)
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
    class_indices = {'Impressionism': 0, 'Realisme': 1, 'Expressionism': 2}
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


# Function to simulate model retraining (replace with your actual model retraining logic)
def retrain_model(update_image_path):
    # Placeholder logic, replace with your actual model retraining code
    # For example, you can use transfer learning on the new image data
    # or fine-tune an existing model
    print(f"Retraining model with new image: {update_image_path}")
    # Your custom retraining logic here




if __name__ == "__main__":
    app.run()

