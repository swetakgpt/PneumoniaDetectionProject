# app.py
import os
import random
import pickle
from datetime import datetime
import shutil # For robust file deletion

import cv2 # OpenCV for image processing
import numpy as np
from bson import ObjectId
from dotenv import load_dotenv
from flask import (Flask, render_template, request, session,
                   redirect, url_for, flash)
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from flask_wtf import CSRFProtect # For CSRF Protection
from flask_wtf.file import FileField, FileRequired, FileAllowed # For form file validation (optional here as we do it manually)

import numpy as np
from tensorflow.keras.models import load_model

# --- Configuration ---
load_dotenv(dotenv_path="mongo_cred.env")

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY')
if not app.secret_key:
    raise ValueError("FLASK_SECRET_KEY environment variable not set. Please set it in mongo_cred.env")

csrf = CSRFProtect(app) # Initialize CSRF protection

# File Upload Configuration
UPLOAD_FOLDER = 'uploads_temp' # Temporary storage for uploaded images before prediction
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# MongoDB Connection
MONGO_URI = os.getenv('MONGO_URI')
if not MONGO_URI:
    raise ValueError("MONGO_URI environment variable is not set")

try:
    client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
    client.admin.command('ping')
    print("Successfully connected to MongoDB!")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    # Consider how to handle this: exit, or run with limited functionality
    # For now, we'll let it proceed and routes will fail if DB is needed.

db = client['pneumonia_detector_db'] # Database name
users_collection = db['users']
medicines_collection = db['medicines']
predictions_collection = db['predictions'] # For prediction history

# # Load the pre-trained model
# MODEL_PATH = 'model.pkl'
# loaded_model = None
# if os.path.exists(MODEL_PATH):
#     try:
#         with open(MODEL_PATH, 'rb') as file:
#             loaded_model = pickle.load(file)
#         print("Model loaded successfully.")
#     except Exception as e:
#         print(f"Error loading model.pkl: {e}")
# else:
#     print(f"Warning: {MODEL_PATH} not found. Prediction functionality will be unavailable.")


# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

MODEL_PATH = 'CNN_Classification_1.h5'
loaded_model = None

if os.path.exists(MODEL_PATH):
    try:
        loaded_model = load_model(MODEL_PATH)
        print("Model loaded successfully from .h5 file.")
    except Exception as e:
        print(f"Error loading {MODEL_PATH}: {e}")
else:
    print(f"Warning: {MODEL_PATH} not found. Prediction functionality will be unavailable.")


def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # model expects grayscale input
        if img is None:
            print(f"Warning: Could not read image at {image_path}")
            return None
        img = cv2.resize(img, (64, 64))  # match model input size
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=(0, -1))  # Shape: (1, 64, 64, 1)
        return img
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None


def predict_image_class(image_path):
    if loaded_model is None:
        print("Model not loaded. Cannot predict.")
        return None, None

    processed_img = preprocess_image(image_path)
    if processed_img is None:
        return None, None

    try:
        prediction = loaded_model.predict(processed_img)
        raw_confidence = float(prediction[0][0])  # sigmoid output

        if raw_confidence > 0.5:
            predicted_class_label = "Pneumonia"
            display_confidence = raw_confidence
        else:
            predicted_class_label = "Normal"
            display_confidence = 1.0 - raw_confidence

        return predicted_class_label, display_confidence
    except Exception as e:
        print(f"Error during model prediction: {e}")
        return None, None

# --- Decorators ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

# --- Context Processors ---
@app.context_processor
def inject_global_vars():
    user_info = None
    if 'user_id' in session:
        user = users_collection.find_one({'_id': ObjectId(session['user_id'])})
        if user: # Make sure user still exists
            user_info = {
                '_id': str(user['_id']),
                'username': user.get('username'),
                'displayName': user.get('displayName')
            }
        else: # User in session doesn't exist in DB, clear session
            session.clear()
            flash("Your session was invalid. Please log in again.", "warning")
            # This redirect won't work directly inside a context processor.
            # The effect is that user_info will be None, and @login_required will catch it.

    return dict(current_user_info=user_info, G_YEAR=datetime.now().year)

# --- Error Handlers ---
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    # It's good practice to log the actual error here for debugging
    app.logger.error(f"Server Error: {e}", exc_info=True)
    flash("An unexpected error occurred on our end. Please try again later.", "danger")
    return render_template('500.html'), 500

# --- Routes ---
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        if not username or not password:
            flash('Username and password are required.', 'danger')
            return render_template('register.html')
        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'warning')
            return render_template('register.html')

        existing_user = users_collection.find_one({'username_lower': username.lower()}) # Case-insensitive check
        if existing_user:
            flash('Username already exists. Please choose another.', 'warning')
            return render_template('register.html')

        hashed_password = generate_password_hash(password)
        new_user = {
            'username': username,
            'username_lower': username.lower(), # For case-insensitive lookup
            'password': hashed_password,
            'createdAt': datetime.utcnow()
        }
        users_collection.insert_one(new_user)
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        if not username or not password:
            flash('Username and password are required.', 'danger')
            return render_template('login.html')

        user = users_collection.find_one({'username_lower': username.lower()}) # Case-insensitive login
        if user and check_password_hash(user['password'], password):
            session['user_id'] = str(user['_id'])
            # session['username'] is handled by context_processor (current_user_info)
            flash(f'Welcome back, {user["username"]}!', 'success')
            next_url = request.args.get('next')
            return redirect(next_url or url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'danger')
            return render_template('login.html')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict_scan', methods=['GET', 'POST']) # Renamed /index to /predict_scan
@login_required
def predict_scan():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash("No file part in the request.", "warning")
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash("No file selected.", "warning")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            original_filename = secure_filename(file.filename)
            # Create a unique filename for temp storage to avoid collisions
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            unique_filename = f"{timestamp}_{original_filename}"
            temp_image_path = os.path.join(UPLOAD_FOLDER, unique_filename)
            
            try:
                file.save(temp_image_path)
                prediction_result_label, confidence = predict_image_class(temp_image_path)

                if prediction_result_label is None:
                    flash("Could not process the image or make a prediction.", "danger")
                    return redirect(request.url)
                
                # Store prediction history
                history_entry = {
                    'user_id': ObjectId(session['user_id']),
                    'original_filename': original_filename,
                    'prediction_result': prediction_result_label,
                    'confidence': confidence,
                    'predicted_at': datetime.utcnow()
                }
                predictions_collection.insert_one(history_entry)
                
                # Pass result and confidence to the result page
                # No need to pass filename as we are not displaying the image anymore
                session['last_prediction_result'] = prediction_result_label
                session['last_prediction_confidence'] = f"{confidence:.2%}" if confidence is not None else "N/A"
                return redirect(url_for('prediction_result_page'))

            except Exception as e:
                app.logger.error(f"Error during prediction or file handling: {e}", exc_info=True)
                flash("An error occurred during the prediction process.", "danger")
                return redirect(request.url)
            finally:
                # Securely delete the temporary file
                if os.path.exists(temp_image_path):
                    try:
                        os.remove(temp_image_path)
                    except OSError as e:
                        app.logger.error(f"Error deleting temp file {temp_image_path}: {e}")
        else:
            flash(f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}.", "danger")
            return redirect(request.url)
            
    return render_template('index.html') # This is the upload page

@app.route('/prediction_result')
@login_required
def prediction_result_page():
    result = session.pop('last_prediction_result', None)
    confidence = session.pop('last_prediction_confidence', 'N/A')
    if result is None:
        flash("No prediction result found. Please try predicting again.", "info")
        return redirect(url_for('predict_scan'))
    return render_template('result.html', result=result, confidence=confidence)


@app.route("/find_doctors")
@login_required
def find_doctors():
    # This could be expanded to fetch from a DB or an API
    doctors = [
        {"name": "Dr. S K Agrawal", "speciality": "Chest", "address": "Maurya Bhawan, Varanasi", "phone": "+91 XXXXXX"},
        {"name": "Dr. Samaria", "speciality": "Multi-speciality & Chest", "address": "Durgakund, Varanasi", "phone": "+91 YYYYYY"},
        {"name": "Dr. Manoj Kumar Gupta", "speciality": "Chest & Respiratory", "address": "Lanka, Varanasi", "phone": "+91 ZZZZZZ"},
    ]
    random_doctor = random.choice(doctors)
    return render_template("find_doctors.html", doctor=random_doctor)

@app.route("/profile", methods=['GET', 'POST'])
@login_required
def profile():
    user_object_id = ObjectId(session['user_id'])
    if request.method == 'POST':
        displayName = request.form.get('displayName', '').strip()
        medicalHistory = request.form.get('medicalHistory', '').strip()
        
        update_fields = {}
        # Only update if new value is provided, or allow clearing if empty string is submitted
        current_user = users_collection.find_one({'_id': user_object_id})
        if displayName != current_user.get('displayName', ''):
             update_fields['displayName'] = displayName
        if medicalHistory != current_user.get('medicalHistory', ''):
            update_fields['medicalHistory'] = medicalHistory
        
        if update_fields:
            users_collection.update_one(
                {'_id': user_object_id},
                {'$set': update_fields}
            )
            flash('Profile details updated successfully!', 'success')
        else:
            flash('No changes submitted.', 'info')
        return redirect(url_for('profile'))

    user_data = users_collection.find_one({'_id': user_object_id})
    return render_template('profile.html', 
                           displayName=user_data.get('displayName', ''),
                           medicalHistory=user_data.get('medicalHistory', ''))

@app.route('/add_medicine', methods=['GET', 'POST'])
@login_required
def add_medicine():
    if request.method == 'POST':
        medicine_name = request.form.get('medicineName', '').strip()
        repetitiveness = request.form.get('repetitiveness')
        repetition_count_str = request.form.get('repetitionCount')
        medicine_notes = request.form.get('medicineNotes', '').strip()

        if not medicine_name or not repetitiveness or not repetition_count_str:
            flash('Medicine Name, Frequency, and Count are required.', 'danger')
            return render_template('add_medicine.html', form_data=request.form) # Pass back form data

        try:
            repetition_count = int(repetition_count_str)
            if repetition_count <= 0:
                raise ValueError()
        except ValueError:
            flash('Repetition count must be a valid positive number.', 'danger')
            return render_template('add_medicine.html', form_data=request.form)

        medicine_data = {
            'user_id': ObjectId(session['user_id']),
            'medicine_name': medicine_name,
            'repetitiveness': repetitiveness,
            'repetition_count': repetition_count,
            'medicineNotes': medicine_notes,
            'added_at': datetime.utcnow()
        }
        medicines_collection.insert_one(medicine_data)
        flash(f"Medicine '{medicine_name}' added to your schedule!", 'success')
        return redirect(url_for('medicine_report'))
            
    return render_template('add_medicine.html', form_data={})

@app.route('/medicine_report')
@login_required
def medicine_report():
    user_object_id = ObjectId(session['user_id'])
    user_medicines = list(medicines_collection.find({'user_id': user_object_id}).sort("added_at", -1))
    return render_template('medicine_report.html', medicines=user_medicines)

@app.route('/prediction_history')
@login_required
def prediction_history():
    user_object_id = ObjectId(session['user_id'])
    history = list(predictions_collection.find({'user_id': user_object_id}).sort("predicted_at", -1))
    return render_template('prediction_history.html', history=history)

# # --- Main Application Runner ---
# if __name__ == '__main__':
#     # For production, use a proper WSGI server like Gunicorn or Waitress.
#     # Example: gunicorn -w 4 -b 0.0.0.0:5000 app:app
#     # For development:
#     app.run(debug=True, host='0.0.0.0', port=5000)

# app.py

# ... (all your Flask app code) ...

# --- Main Application Runner ---
if __name__ == '__main__':
    # For local development ONLY, Gunicorn will run it in production
    # You might want to get the port from an environment variable for flexibility
    port = int(os.environ.get("PORT", 5000)) # Render often sets PORT env var
    app.run(debug=False, host='0.0.0.0', port=port) # Set debug=False for local testing similar to prod