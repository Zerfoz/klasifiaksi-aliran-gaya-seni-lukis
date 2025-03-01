import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from flask_bcrypt import Bcrypt
from connect import mydbcon
from flask import Flask, render_template, flash, request, redirect, url_for, jsonify, session
from PIL import Image
from flask_cors import CORS
from tensorflow.keras.preprocessing.image import ImageDataGenerator


app = Flask(__name__)
app.secret_key = "my apps super"
bcrypt = Bcrypt(app)

# Folder untuk menyimpan gambar yang diunggah
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')


def login(username, password):
    mycursor = mydbcon.cursor()
    sql = "SELECT password FROM user WHERE username = %s"
    val = (username,)
    mycursor.execute(sql, val)
    result = mycursor.fetchone()

    if result:
        hashed_password = result[0]
        # Check password
        if bcrypt.check_password_hash(hashed_password, password):
            return True
    return False


@app.route('/login', methods=['GET', 'POST'])
def user_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if login(username, password):
            session['username'] = username
            # Mengecek apakah user adalah admin atau bukan
            if is_admin(username):
                return redirect(url_for('index_user'))
            else:
                return redirect(url_for('index_adm'))
        else:
            flash('Invalid username or password')
    return render_template('login.html')

# Fungsi untuk memeriksa apakah username adalah admin
def is_admin(username):
    mycursor = mydbcon.cursor()
    sql = "SELECT COUNT(*) FROM admin WHERE username = %s"
    val = (username,)
    mycursor.execute(sql, val)
    result = mycursor.fetchone()
    return result[0] > 0  # Jika result lebih dari 0 berarti admin, jika tidak berarti user biasa

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if login(username, password):
            session['username'] = username
            return redirect(url_for('index_adm'))
        else:
            flash('Invalid username or password')
    return render_template('admin/dashboard_admin.html')  # Rute login untuk admin terpisah

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out.')
    return redirect(url_for('user_login'))  # Logout mengarah ke login user

@app.route('/input_user', methods=['GET', 'POST'])
def input_user():
    if request.method == 'POST':
        # Mengambil data dari form
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # Simpan data ke dalam database (bisa ditambahkan validasi dan enkripsi password)
        mycursor = mydbcon.cursor()
        sql = "INSERT INTO user (username, email, password) VALUES (%s, %s, %s)"
        val = (username, email, bcrypt.generate_password_hash(password).decode('utf-8'))  # Enkripsi password
        mycursor.execute(sql, val)
        mydbcon.commit()

        flash('User berhasil ditambahkan!', 'success')
        return redirect(url_for('user_login'))  # Redirect ke halaman utama setelah berhasil

    return render_template('input_user.html')  # Menampilkan form input


@app.route('/index_user')
def index_user():
    if 'username' in session:
        username = session['username']  # Ambil username dari session
        return render_template('user/dashboard_user.html', username=username)
    else:
        return redirect(url_for('user_login'))

@app.route('/index_adm')
def index_adm():
    if 'username' in session:
        username = session['username']  # Ambil username dari session
        return render_template('admin/dashboard_admin.html', username=username)
    else:
        return redirect(url_for('admin_login'))

@app.route('/deteksi')
def deteksi():
    return render_template('deteksi.html')

@app.route('/deteksi_admin')
def deteksi_admin():
    return render_template('admin/deteksi_admin.html')

@app.route('/deteksi_user')
def deteksi_user():
    return render_template('user/deteksi_user.html')

@app.route('/data')
def data():
    return render_template('data.html')

@app.route('/datalatih')
def datalatih():
    return render_template('admin/datalatih.html')

@app.route('/datalatih_user')
def datalatih_user():
    return render_template('user/datalatih_user.html')

@app.route('/tentang')
def tentang():
    return render_template('tentang.html')

@app.route('/tentang_adm')
def tentang_adm():
    return render_template('admin/tentang_adm.html')


@app.route('/lukisan', methods=['POST'])
def lukisan_classifier():
    try:
        # Ambil gambar yang dikirim melalui form
        image_request = request.files['image']
        image_pil = Image.open(image_request)
        
        # Konversi gambar ke RGB (jika tidak dalam format RGB)
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')
        
        # Resize gambar ke ukuran yang diharapkan
        expected_size = (224, 224)
        resized_image_pil = image_pil.resize(expected_size)
        
        # Konversi gambar ke array numpy dan normalisasi
        image_array = np.array(resized_image_pil)
        
        # Pastikan gambar memiliki 3 channel
        if image_array.shape[-1] != 3:
            return jsonify({'error': 'Gambar harus memiliki 3 channel RGB.'}), 400

        # Normalisasi nilai pixel
        rescaled_image_array = image_array / 255.0
        batched_rescaled_image_array = np.expand_dims(rescaled_image_array, axis=0)

        # Load tiga model
        model1 = tf.keras.models.load_model("keras_model_1.h5")
        model2 = tf.keras.models.load_model("keras_model.h5")
        model3 = tf.keras.models.load_model("keras_model_3.h5")
        
        # Prediksi menggunakan ketiga model
        result1 = model1.predict(batched_rescaled_image_array)
        result2 = model2.predict(batched_rescaled_image_array)
        result3 = model3.predict(batched_rescaled_image_array)

        # Kombinasikan hasil prediksi dari ketiga model
        combined_result = (result1 + result2 + result3) / 3  # Rata-rata prediksi dari ketiga model
        
        # Format hasil prediksi
        formatted_result = get_formated_predict_result(combined_result)
        
        print(f"Prediksi berhasil: {formatted_result}")  # Logging

        # Return hasil prediksi dalam format JSON
        return jsonify(formatted_result), 200

    except Exception as e:
        print(f"Error: {e}")  # Logging error
        return jsonify({'error': f'Terjadi kesalahan saat memproses gambar: {str(e)}'}), 500




from PIL import Image
import io

@app.route('/save_lukisan', methods=['POST'])
def save_lukisan():
    if 'image' not in request.files:
        return jsonify({'message': 'No image part'}), 400

    image = request.files['image']
    class_name = request.form['class']
    confidence = request.form['confidence']
    date = request.form['date']
    new_filename = request.form['file_name'] + '.jpg'  # Tambahkan .jpg ke nama file

    if image.filename == '':
        return jsonify({'message': 'No selected image'}), 400

    # Membaca gambar dan mengonversinya ke format JPG
    image_pil = Image.open(image)
    image_jpg = io.BytesIO()
    image_pil.convert('RGB').save(image_jpg, format='JPEG')
    image_jpg.seek(0)

    # Buat folder berdasarkan aliran seni
    class_folder = os.path.join(app.config['UPLOAD_FOLDER'], class_name)
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)

    filepath = os.path.join(class_folder, new_filename)

    # Simpan gambar sebagai file JPG
    with open(filepath, 'wb') as f:
        f.write(image_jpg.getvalue())

    # Simpan informasi di database
    mycursor = mydbcon.cursor()
    sql = "INSERT INTO prediksi (nama_file, aliran, confidence, tanggal) VALUES (%s, %s, %s, %s)"
    val = (new_filename, class_name, confidence, date)
    mycursor.execute(sql, val)
    mydbcon.commit()

    return jsonify({'message': 'Data saved successfully'}), 200



def get_formated_predict_result(predict_result):
    class_indices = {
        'Bukan citra lukisan': 0,
        'Expressionism': 1,
        'Impressionism': 2,
        'Realism': 3,
        'Renaissance': 4,
        'Romanticism': 5
    }
    # descriptions = {
    #     'Bukan citra lukisan': 'Gambar ini bukan merupakan citra lukisan.',
    #     'Expressionism': 'Expressionism adalah aliran seni yang mengekspresikan emosi dan perasaan seniman.',
    #     'Impressionism': 'Impressionism adalah aliran seni yang menekankan pada efek cahaya dan warna.',
    #     'Realism': 'Realism adalah aliran seni yang menggambarkan subjek dengan cara yang realistis.',
    #     'Renaissance': 'Renaissance adalah aliran seni yang menekankan pada kebangkitan kembali budaya klasik dan pemahaman ilmiah.',
    #     'Romanticism': 'Romanticism adalah aliran seni yang menekankan pada emosi, individualisme, dan glorifikasi masa lalu.'
    # }
    descriptions = {
        'Bukan citra lukisan': 'Gambar ini bukan merupakan citra lukisan.',
        'Expressionism': 'Expressionism adalah aliran seni yang menekankan ekspresi emosi dan suasana hati melalui bentuk dan warna yang sering terdistorsi.',
        'Impressionism': 'Impressionism adalah aliran seni yang berfokus pada efek cahaya dan warna untuk menangkap kesan sesaat suatu momen.',
        'Realism': 'Realism adalah aliran seni yang menggambarkan subjek dengan cara yang realistis.',
        'Renaissance': 'Renaissance adalah aliran seni yang menggabungkan perspektif ilmiah dengan tema-tema klasik, menonjolkan harmoni dan keindahan setiap objek.',
        'Romanticism': 'Romanticism adalah aliran seni yang menonjolkan emosi kuat, keindahan alam yang dramatis, dan cerita atau tema yang heroik atau melankolis.'
    }
    inverted_class_indices = {v: k for k, v in class_indices.items()}

    processed_predict_result = predict_result[0]
    maxIndex = 0
    maxValue = 0

    for index in range(len(processed_predict_result)):
        if processed_predict_result[index] > maxValue:
            maxValue = processed_predict_result[index]
            maxIndex = index

    predicted_class = inverted_class_indices[maxIndex]
    confidence = maxValue * 100
    description = descriptions[predicted_class]

    # Tambahkan persentase untuk setiap kelas
    all_classes_with_confidence = [
        {
            'class': inverted_class_indices[i],
            'confidence': round(processed_predict_result[i] * 100, 2)
        }
        for i in range(len(processed_predict_result))
    ]

    return {
        'class': predicted_class,
        'confidence': confidence,
        'description': description,
        'all_classes': all_classes_with_confidence  # Semua kelas dengan persentase
    }



# Function to simulate model retraining (replace with your actual model retraining logic)
def retrain_model(update_image_path):
    # Placeholder logic, replace with your actual model retraining code
    # For example, you can use transfer learning on the new image data
    # or fine-tune an existing model
    print(f"Retraining model with new image: {update_image_path}")
    # Your custom retraining logic here

@app.route('/get_paintings/<aliran>', methods=['GET'])
def get_paintings(aliran):
    mycursor = mydbcon.cursor()
    mycursor.execute("SELECT id_prediksi, nama_file, confidence, tanggal, aliran FROM prediksi WHERE aliran = %s", (aliran,))
    paintings = mycursor.fetchall()
    mycursor.close()
    
    paintings_list = []
    for painting in paintings:
        paintings_list.append({
            'id_prediksi': painting[0],
            'nama_file': painting[1],
            'confidence': painting[2],
            'tanggal': painting[3].strftime('%Y-%m-%d') if painting[3] else None,
            'aliran': painting[4]
        })
    
    return jsonify(paintings_list)


@app.route('/update_model', methods=['POST'])
def update_model():
    try:
        # Path ke dataset
        dataset_dir = 'static/uploads'
        # Parameter pelatihan
        batch_size = 32
        img_height = 224
        img_width = 224

        # ImageDataGenerator untuk preprocessing dan augmentasi
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.15  # Menggunakan 20% data untuk validasi
        )

        # Menghasilkan data latih
        train_generator = train_datagen.flow_from_directory(
            dataset_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )

        # Menghasilkan data validasi
        validation_generator = train_datagen.flow_from_directory(
            dataset_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )

        # Buat model
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(len(train_generator.class_indices), activation='softmax')  # Jumlah kelas
        ])

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Latih model
        history = model.fit(
            train_generator,
            epochs=10,
            validation_data=validation_generator
        )

        # Simpan model
        model.save('new_keras_model.h5')

        print("Model berhasil dilatih dan disimpan sebagai 'new_keras_model.h5'")
        return jsonify({"message": "Model berhasil diperbarui!"}), 200
    
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")
        return jsonify({"message": "Terjadi kesalahan saat memperbarui model."}), 500




if __name__ == '__main__':
    app.run(debug=True, port=8080)
