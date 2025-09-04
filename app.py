from flask import Flask, flash, render_template, request, redirect, url_for, session, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from random import randint
from werkzeug.utils import secure_filename
from flask_bcrypt import Bcrypt
from keras.models import load_model
from PIL import Image, ImageChops, ImageEnhance
import cv2
import time
import secrets
from PIL.ExifTags import TAGS
import os
from flask import send_file, abort
from fpdf import FPDF
import random
import numpy as np
from datetime import datetime
from flask_migrate import Migrate
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash
from itsdangerous import URLSafeSerializer
import string



# Initialize Flask app
app = Flask(__name__, static_folder='static')

# Define the persistent data path for Render
DATA_DIR = '/mnt/data'
DB_PATH = os.path.join(DATA_DIR, 'database.db')
UPLOAD_FOLDER = os.path.join(DATA_DIR, 'uploads')

# Use the default paths if not on Render (for local testing)
if not os.path.exists(DATA_DIR):
    DB_PATH = 'sqlite:///database.db'
    UPLOAD_FOLDER = "uploads"
else:
    DB_PATH = f'sqlite:///{DB_PATH}'

app.config['SQLALCHEMY_DATABASE_URI'] = DB_PATH
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-default-secret-key')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# Initialize Database
db = SQLAlchemy(app)

serializer = URLSafeSerializer(app.config['SECRET_KEY'])
bcrypt=Bcrypt(app)


# Initialize Flask-Migrate
migrate = Migrate(app, db)
#otp generation
otp = randint(100000, 999999)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Load the trained model
try:
    model = load_model('forgery_model.h5')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# User Model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(80), nullable=False, unique=True)
    mobile = db.Column(db.String(10), nullable=False, unique=True)
    password = db.Column(db.String(60), nullable=False)
    reset_token = db.Column(db.String(100), nullable=True)
    analyses = db.relationship('AnalysisHistory', backref='user', lazy=True)

# AnalysisHistory Model
class AnalysisHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_name = db.Column(db.String(120), nullable=False)
    result = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    image_metadata = db.Column(db.Text, nullable=True)

# Create tables (if they don't exist)
with app.app_context():
    db.create_all()
    print("Tables created successfully!")
    
# Email Config
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'imageauthentix@gmail.com'
app.config['MAIL_PASSWORD'] = 'pajh gwns tbis qndt'
app.config['MAIL_DEFAULT_SENDER'] = 'imageauthentix@gmail.com'
mail = Mail(app)

@app.route('/forgot-password', methods=['GET', 'POST']) 
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        user = User.query.filter_by(email=email).first()
        errors = []
        if user:
            # Generate a unique token
            reset_token = secrets.token_urlsafe(16)
            user.reset_token = reset_token
            db.session.commit()

            # Generate reset URL
            reset_url = url_for('reset_password', token=reset_token, _external=True)

            # Send reset email
            msg = Message('Password Reset Request', recipients=[email])
            msg.body = f'Please click the link to reset your password: {reset_url}'
            mail.send(msg)
            errors.append('Mail has been sent to reset your password!')
            return render_template('forgot_password.html', errors=errors)


        else:
            errors.append('Email not found!')
            return render_template('forgot_password.html', errors=errors)
        # return redirect(url_for('login'))
    return render_template('forgot_password.html')

# Reset Password Route
@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    user = User.query.filter_by(reset_token=token).first()
    if not user:
        flash('Invalid User', 'danger')
        return redirect(url_for('login'))

    if request.method == 'POST':
        new_password = request.form['new-password']
        confirm_password = request.form['confirm-password']
        errors = []
        if new_password != confirm_password:
            errors.append('Password does not match')
            return render_template('reset_password.html', errors=errors, token=token)

        user.password = bcrypt.generate_password_hash(new_password)
        user.reset_token = None  # Clear the token after reset
        db.session.commit()

        errors.append("Your password has been reset successfully. Please log in.")
        return render_template('login.html', errors=errors)
    return render_template('reset_password.html', token=token)

# User loader for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Home Route
@app.route('/')
def home():
    return render_template('index.html')

# Signup Route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['name'].strip()
        email = request.form['email'].strip()
        mobile = request.form['mobile'].strip()
        password = request.form['password'].strip()
        confirm_password = request.form['confirmpassword'].strip()

        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('signup'))

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email already exists! Please use a different email.', 'danger')
            return redirect(url_for('signup'))

        # Use generate_password_hash instead of hashpw
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(username=username, email=email, mobile=mobile, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Signup successful! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('signup.html')



# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password'].encode('utf-8')
        user = User.query.filter_by(email=email).first()

        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            flash('Login successful!', 'success')
            token = serializer.dumps(email)
            return redirect(url_for('verify_email', token=token))
        else:
            flash('Invalid credentials. Please try again.', 'danger')
    
    return render_template('login.html')

@app.route('/verify_email', methods=['GET'])
def verify_email():
    token = request.args.get('token')
    if token:
        try:
            email = serializer.loads(token)
            session['email'] = email
            # Generate a new OTP every time the token is present (i.e., after login)
            otp = randint(100000, 999999)
            session['otp'] = otp
            msg = Message('OTP', recipients=[email])
            msg.body = f'Your OTP is {otp}'
            mail.send(msg)
        except:
            flash('Invalid or expired token.', 'danger')
            return redirect(url_for('login'))
    
    email = session.get('email')
    if not email:
        flash('Please log in to access this page.', 'danger')
        return redirect(url_for('login'))
    
    user = User.query.filter_by(email=email).first()
    if not user:
        flash("User not found.", "danger")
        return redirect(url_for('signup'))

    return render_template('verify_email.html', username=user.username)

@app.route('/validate', methods=['POST'])
def validate():
    user_otp = request.form.get('otp', '').strip()
    errors = []

    # Check if the user input is numeric
    if not user_otp.isdigit():
        errors.append('Invalid OTP format. Please enter only numeric digits.')
        return render_template("verify_email.html", errors=errors)

    # Compare user OTP to the one stored in the session
    if session.get('otp') and int(user_otp) == session['otp']:
        errors.append('Email is verified successfully!')
        return render_template('integrate.html')
    else:
        errors.append('Incorrect OTP. Please try again.')
        # Just reload the page with an error message, without resending the email
        return render_template('verify_email.html', errors=errors)


@app.route('/resend', methods=['GET'])
def resend():
    # Generate a new OTP
    new_otp = randint(100000, 999999)
    session['otp'] = new_otp
    email = session.get('email')

    # Send the new OTP email
    msg = Message('OTP', recipients=[email])
    msg.body = f'Your new OTP is {new_otp}'
    mail.send(msg)

    # Inform the user that a new OTP has been sent
    errors = ['A new OTP has been sent to your email.']
    user = User.query.filter_by(email=email).first()
    return render_template('verify_email.html', username=user.username, errors=errors)

# Dashboard Route
@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('integrate.html', user=current_user)

# Contact Route
@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        firstname = request.form["firstname"]
        lastname = request.form["lastname"]
        email = request.form["email"]
        company = request.form.get("company", "N/A")  # Optional field
        message = request.form["message"]

        # Here, you can add logic to send this message via email or save it to a database.
        print(f"Received message from {firstname} {lastname} ({email}) - {company}: {message}")

        # Flash success message
        flash("Your message has been sent successfully!", "success")
        return redirect(url_for("contact"))

    return render_template("contact.html")

# About Route
@app.route('/about')
def about():
    return render_template('about.html')

# Analyze Route
@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"✅ File saved at: {filepath}")

        # Check if the model is loaded
        if model is None:
            return jsonify({"error": "Model not loaded. Please check the logs."}), 500

        # Perform forgery detection
        class_label, confidence, ela_image = detect_forgery(filepath)
        ela_variance_label, ela_variance_value = calculate_ela_variance(ela_image)
        noise_consistency_label, noise_consistency_value = calculate_noise_consistency(ela_image)
        size, format, mode, file_size = get_image_metadata(filepath)

        # Get additional metadata
        creation_date, modification_date = get_image_creation_date(filepath)
        camera_info = get_camera_info(filepath)
        location_info = get_location_info(filepath)
        # Validate metadata
        validation_results = validate_metadata(creation_date, modification_date, camera_info, location_info)

       # Format metadata with HTML line breaks
        metadata = f"""
        Size: {size[0]}x{size[1]}<br>
        Format: {format}<br>
        Mode: {mode}<br>
        File Size: {file_size}<br>
        Creation Date: {creation_date}<br>
        Modification Date: {modification_date}<br>
        Camera Make: {camera_info['Make']}<br>
        Camera Model: {camera_info['Model']}<br>
        Software: {camera_info['Software']}<br>
        Aperture: {camera_info['Aperture']}<br>
        Shutter Speed: {camera_info['ShutterSpeed']}<br>
        ISO: {camera_info['ISO']}<br>
        Focal Length: {camera_info['FocalLength']}<br>
        Orientation: {camera_info['Orientation']}<br>
        Location: {location_info['Latitude']}, {location_info['Longitude']} (Timestamp: {location_info['Timestamp']})<br>
        <br>
        Validation Results:<br>
        - Timestamp Mismatch: {validation_results['Timestamp Mismatch']}<br>
        - GPS Data Valid: {validation_results['GPS Data Valid']}<br>
        - Editing Software: {validation_results['Editing Software']}
        """
        # Save the highlighted image path if the image is forged
        highlighted_image_filename = None
        if class_label == "Forged":
            highlighted_image_filename = highlight_fake_regions(filepath)
            print(f"✅ Highlighted image saved as: {highlighted_image_filename}")

        # Save analysis history to the database
        analysis = AnalysisHistory(
            user_id=current_user.id,
            image_name=highlighted_image_filename if highlighted_image_filename else filename,
            result=class_label,
            confidence=confidence,
            image_metadata=metadata
        )
        db.session.add(analysis)
        db.session.commit()

        # Prepare response data
        response_data = {
            'class_label': class_label,
            'confidence': f"{confidence * 100:.2f}%",
            'ela_variance': f"{ela_variance_label} (Variance: {ela_variance_value:.2f})",
            'noise_consistency': f"{noise_consistency_label} (Std Dev: {noise_consistency_value:.2f})",
            'metadata': metadata,
            'image': highlighted_image_filename if highlighted_image_filename else None
        }

        return jsonify(response_data)

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        # Delete the original file if it's not needed
        if os.path.exists(filepath) and filepath != os.path.join(app.config['UPLOAD_FOLDER'], highlighted_image_filename if highlighted_image_filename else filename):
            os.remove(filepath)
            print(f"✅ File deleted: {filepath}")

# Reports History Route
@app.route('/reports-history')
@login_required
def reports_history():
    """
    Route to display the reports history page.
    """
    # Fetch analysis history for the current user
    analyses = AnalysisHistory.query.filter_by(user_id=current_user.id).order_by(AnalysisHistory.timestamp.desc()).all()
    
    # Debug: Print the fetched analyses
    print(f"Fetched analyses for user {current_user.id}: {analyses}")
    
    # Pass the data to the template
    return render_template('reports-history.html', analyses=analyses)


@app.route('/get-analysis-history')
@login_required
def get_analysis_history():
    """
    Route to fetch analysis history for the current user.
    """
    # Fetch analysis history for the current user
    analyses = AnalysisHistory.query.filter_by(user_id=current_user.id).order_by(AnalysisHistory.timestamp.desc()).all()
    
    # Prepare response data
    analysis_history = []
    for analysis in analyses:
        analysis_history.append({
            'id': analysis.id,  # Ensure this field is included
            'image_name': analysis.image_name,
            'result': analysis.result,
            'confidence': analysis.confidence,
            'timestamp': analysis.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'image_metadata': analysis.image_metadata,
            'image': analysis.image_name  # Include the image name in the response
        })
    
    return jsonify(analysis_history)

@app.route('/clear-analysis-history', methods=['POST'])
@login_required
def clear_analysis_history():
    """
    Route to clear the analysis history for the current user.
    """
    try:
        # Delete all analysis history for the current user
        AnalysisHistory.query.filter_by(user_id=current_user.id).delete()
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500
# Logout Route
@app.route('/logout')
@login_required
def logout():
    """
    Route to handle user logout.
    """
    logout_user()
    return redirect(url_for('home'))  # Redirect to the home page after logout

# Uploaded Files Route
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Helper Functions for Image Analysis
# def convert_to_ela_image(path, quality=90):
#     temp_filename = 'temp_file_name.jpg'
#     image = Image.open(path).convert('RGB')
#     image.save(temp_filename, 'JPEG', quality=quality)
#     temp_image = Image.open(temp_filename)
#     ela_image = ImageChops.difference(image, temp_image)
#     extrema = ela_image.getextrema()
#     max_diff = max([ex[1] for ex in extrema])
#     scale = 255.0 / max_diff if max_diff else 1
#     ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
#     os.remove(temp_filename)
#     return ela_image

def prepare_image(image_path, image_size=(128, 128)):
    ela_image = convert_to_ela_image(image_path)
    ela_image = ela_image.resize(image_size)
    ela_array = np.array(ela_image).astype('float32') / 255.0  # Normalize
    ela_array = np.expand_dims(ela_array, axis=0)  # Reshape for model
    return ela_image, ela_array

def detect_forgery(image_path):
    ela_image, image_array = prepare_image(image_path)
    prediction = model.predict(image_array)[0]
    class_label = "Authentic" if np.argmax(prediction) == 1 else "Forged"
    confidence = np.max(prediction)
    return class_label, confidence, ela_image

def highlight_fake_regions(image_path):
    ela_image, _ = prepare_image(image_path)
    ela_gray = np.array(ela_image.convert('L'))
    original = cv2.imread(image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    ela_gray = cv2.resize(ela_gray, (original.shape[1], original.shape[0]))

    # Edge detection with Sobel filter
    sobelx = cv2.Sobel(ela_gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(ela_gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel_edges = np.sqrt(sobelx**2 + sobely**2)
    sobel_edges = cv2.normalize(sobel_edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Create heatmap and overlay
    heatmap = cv2.applyColorMap(sobel_edges, cv2.COLORMAP_JET)
    overlaid = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    # Generate unique filename using the original filename + timestamp
    timestamp = str(int(time.time()))
    base_name, ext = os.path.splitext(os.path.basename(image_path))  # Extract base name and extension
    output_filename = f"{base_name}_{timestamp}{ext}"  # Original filename + timestamp
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    
    # Save overlaid image with the unique name in the 'uploads' folder
    cv2.imwrite(output_path, cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR))
    return output_filename  # Return only the filename (not the full path)

def calculate_ela_variance(ela_image):
    variance = np.var(np.array(ela_image.convert('L')))
    return ("Low" if variance < 50 else "Medium" if variance < 150 else "High"), variance

def calculate_noise_consistency(ela_image):
    noise_std = np.std(np.array(ela_image.convert('L')))
    return ("Low" if noise_std < 30 else "Medium" if noise_std < 70 else "High"), noise_std

def get_image_metadata(image_path):
    img = Image.open(image_path)
    size, format, mode = img.size, img.format, img.mode
    file_size = os.path.getsize(image_path) / 1024  # KB
    return size, format, mode, f"{file_size:.2f} KB"

def get_image_creation_date(image_path):
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        
        creation_date = "N/A"
        modification_date = "N/A"
        
        if exif_data and isinstance(exif_data, dict):
            for tag, value in exif_data.items():
                tag_name = TAGS.get(tag, tag)
                if tag_name == 'DateTime':
                    creation_date = value
                elif tag_name == 'DateTimeOriginal':
                    modification_date = value
        
        return creation_date, modification_date
    except Exception as e:
        print(f"Error extracting creation date: {e}")
        return "N/A", "N/A"
    
def get_camera_info(image_path):
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        
        camera_info = {
            'Make': 'N/A',
            'Model': 'N/A',
            'Software': 'N/A',
            'Aperture': 'N/A',
            'ShutterSpeed': 'N/A',
            'ISO': 'N/A',
            'FocalLength': 'N/A',
            'Orientation': 'N/A'
        }
        
        if exif_data and isinstance(exif_data, dict):
            for tag, value in exif_data.items():
                tag_name = TAGS.get(tag, tag)
                if tag_name == 'Make':
                    camera_info['Make'] = clean_string(value)
                elif tag_name == 'Model':
                    camera_info['Model'] = clean_string(value)
                elif tag_name == 'Software':
                    camera_info['Software'] = clean_string(value)
                elif tag_name == 'ApertureValue':
                    camera_info['Aperture'] = str(value)
                elif tag_name == 'ShutterSpeedValue':
                    camera_info['ShutterSpeed'] = str(value)
                elif tag_name == 'ISOSpeedRatings':
                    camera_info['ISO'] = str(value)
                elif tag_name == 'FocalLength':
                    camera_info['FocalLength'] = str(value)
                elif tag_name == 'Orientation':
                    orientation_map = {
                        1: 'Horizontal (normal)',
                        3: 'Rotated 180°',
                        6: 'Rotated 90° CW',
                        8: 'Rotated 90° CCW'
                    }
                    camera_info['Orientation'] = orientation_map.get(value, 'N/A')
        
        return camera_info
    except Exception as e:
        print(f"Error extracting camera info: {e}")
        return {
            'Make': 'N/A',
            'Model': 'N/A',
            'Software': 'N/A',
            'Aperture': 'N/A',
            'ShutterSpeed': 'N/A',
            'ISO': 'N/A',
            'FocalLength': 'N/A',
            'Orientation': 'N/A'
        }

def get_location_info(image_path):
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        
        gps_info = {}
        if exif_data and isinstance(exif_data, dict):  # Ensure exif_data is a dictionary
            for tag, value in exif_data.items():
                tag_name = TAGS.get(tag, tag)
                # Look for GPS info in EXIF metadata
                if tag_name == 'GPSInfo' and isinstance(value, dict):  # Ensure GPSInfo is a dictionary
                    gps_info = value
                    break  # Once GPS data is found, exit the loop

        if gps_info:
            # GPSInfo is a dictionary, extract Latitude, Longitude, and Timestamp
            latitude = gps_info.get(2, 'N/A')  # Latitude is at tag 2
            longitude = gps_info.get(4, 'N/A')  # Longitude is at tag 4
            timestamp = gps_info.get(29, 'N/A')  # Timestamp is at tag 29

            # If latitude and longitude are tuples, convert them to decimal degrees
            if latitude != 'N/A' and isinstance(latitude, tuple):
                latitude_decimal = latitude[0] + (latitude[1] / 60.0) + (latitude[2] / 3600.0)
                latitude_dms = decimal_to_dms(latitude_decimal, is_latitude=True)
            else:
                latitude_dms = 'N/A'
            
            if longitude != 'N/A' and isinstance(longitude, tuple):
                longitude_decimal = longitude[0] + (longitude[1] / 60.0) + (longitude[2] / 3600.0)
                longitude_dms = decimal_to_dms(longitude_decimal, is_latitude=False)
            else:
                longitude_dms = 'N/A'
            
            return {
                'Latitude': latitude_dms,
                'Longitude': longitude_dms,
                'Timestamp': timestamp
            }
        else:
            return {
                'Latitude': 'N/A',
                'Longitude': 'N/A',
                'Timestamp': 'N/A'
            }
    except Exception as e:
        print(f"Error extracting location info: {e}")
        return {
            'Latitude': 'N/A',
            'Longitude': 'N/A',
            'Timestamp': 'N/A'
        }
        
def validate_metadata(creation_date, modification_date, camera_info, location_info):
    """
    Validate metadata and generate validation results.
    """
    validation_results = {
        'Timestamp Mismatch': 'No',
        'GPS Data Valid': 'No',
        'Editing Software': 'No'
    }

    # Check for timestamp mismatch
    if creation_date != "N/A" and modification_date != "N/A" and creation_date != modification_date:
        validation_results['Timestamp Mismatch'] = 'Yes'

    # Check for valid GPS data
    if location_info['Latitude'] != 'N/A' and location_info['Longitude'] != 'N/A':
        validation_results['GPS Data Valid'] = 'Yes'

    # Check for editing software
    if camera_info['Software'] != 'N/A':
        validation_results['Editing Software'] = 'Yes'

    return validation_results

def decimal_to_dms(degrees, is_latitude=True):
    """
    Convert decimal degrees to degrees, minutes, and seconds (DMS) format.
    - is_latitude indicates if the degrees are for latitude (North/South)
    """
    # Get absolute value of degrees
    degrees_abs = abs(degrees)
    d = int(degrees_abs)  # Degrees
    m = int((degrees_abs - d) * 60)  # Minutes
    s = (degrees_abs - d - m / 60) * 3600  # Seconds
    
    # Determine the direction (N/S for latitude, E/W for longitude)
    if is_latitude:
        direction = 'N' if degrees >= 0 else 'S'
    else:
        direction = 'E' if degrees >= 0 else 'W'
    
    # Return formatted string
    return f"{d}° {m}' {s:.2f}\" {direction}"

def clean_string(s):
    """
    Clean strings and remove null characters.
    """
    if isinstance(s, bytes):
        s = s.decode('utf-8', errors='ignore')
    return s.strip('\x00')

@app.route('/delete-report/<int:report_id>', methods=['DELETE'])
@login_required
def delete_report(report_id):
    print(f"Received request to delete report with ID: {report_id}")  # Debugging line
    report = AnalysisHistory.query.filter_by(id=report_id, user_id=current_user.id).first()
    
    if not report:
        print(f"Report not found for ID: {report_id}")  # Debugging line
        return jsonify({'success': False, 'error': 'Report not found'}), 404

    try:
        # Get the image path
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], report.image_name)
        
        # Delete the report from the database
        db.session.delete(report)
        db.session.commit()

        # Delete the image file from the filesystem if it exists
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"✅ Image file deleted: {image_path}")

        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        print(f"Error deleting report: {e}")  # Debugging line
        return jsonify({'success': False, 'error': str(e)}), 500
    
    
@app.route('/settings')
def settings():
    return render_template('settings.html')

@app.route('/update_settings', methods=['POST'])
@login_required
def update_settings():
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']

    current_user.username = username
    current_user.email = email
    if password:
        current_user.password = bcrypt.generate_password_hash(password).decode('utf-8')

    db.session.commit()
    return redirect(url_for('settings'))


def convert_to_ela_image(image_path, quality=90):
    """
    Generate an ELA (Error Level Analysis) image from the given image file.

    Args:
        image_path (str): Path to the input image file.
        quality (int): Quality level for re-saving the image (default is 90).

    Returns:
        ela_image (PIL.Image): The generated ELA image.
        ela_filename (str): The filename for the ELA image.
    """
    # Open the original image
    original_image = Image.open(image_path).convert('RGB')

    # Generate a temporary filename for the re-saved image
    temp_filename = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8)) + '.jpg'
    temp_path = os.path.join(os.path.dirname(image_path), temp_filename)

    # Save the original image with the specified quality
    original_image.save(temp_path, 'JPEG', quality=quality)

    # Open the re-saved image
    recompressed_image = Image.open(temp_path)

    # Calculate the difference between the original and re-saved images
    ela_image = ImageChops.difference(original_image, recompressed_image)

    # Enhance the difference to make it more visible
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff if max_diff != 0 else 1
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    # Generate a filename for the ELA image
    ela_filename = 'ela_' + os.path.basename(image_path)

    # Clean up the temporary file
    os.remove(temp_path)

    return ela_image
@app.route('/generate_ela', methods=['POST'])
def generate_ela():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        # Generate ELA image
        ela_image = convert_to_ela_image(file_path)

        # Generate a unique filename for the ELA image
        timestamp = str(int(time.time()))
        ela_filename = f"ela_{timestamp}_{filename}"
        ela_image_path = os.path.join(app.config['UPLOAD_FOLDER'], ela_filename)

        # Save the ELA image to a file
        ela_image.save(ela_image_path)

        # Return the URL of the ELA image
        ela_image_url = f"/uploads/{ela_filename}"

        return jsonify({
            'ela_image_url': ela_image_url,
        })

    except Exception as e:
        print(f"❌ Error generating ELA image: {e}")
        return jsonify({'error': str(e)}), 500

    finally:
        # Clean up the uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"✅ Uploaded file deleted: {file_path}")
@app.route('/ela_image')
@login_required
def ela_image():
    return render_template('ela_image.html')


@app.route('/download_report/<image_name>', methods=['GET'])
@login_required
def download_pdf(image_name):
    # Fetch the report from the database for the current user
    report = AnalysisHistory.query.filter_by(image_name=image_name, user_id=current_user.id).first()
    if not report:
        abort(404, description="Report not found")

    # Create a PDF object
    pdf = FPDF()
    pdf.add_page()

    # Set font for the title
    pdf.set_font("Arial", size=14, style='B')  # Reduced font size for the title
    pdf.cell(200, 10, txt="Image Analysis Report", ln=True, align='C')
    pdf.ln(8)  # Add some space after the title

    # Set font for the content
    pdf.set_font("Arial", size=10)  # Reduced font size for the content

    # Add image name, status, and date
    pdf.cell(200, 8, txt=f"Image Name: {report.image_name}", ln=True)
    pdf.cell(200, 8, txt=f"Status: {report.result.upper()} ({(report.confidence * 100):.2f}%)", ln=True)
    pdf.cell(200, 8, txt=f"Date: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(8)  # Add some space

    # Add metadata (remove <br> tags and format properly)
    metadata = report.image_metadata.replace("<br>", "\n")  # Replace <br> with newlines
    pdf.multi_cell(0, 6, txt=f"Metadata:\n{metadata}")  # Reduced line height for metadata
    pdf.ln(8)  # Add some space

    # Add the highlighted image in the center
    highlighted_image_path = os.path.join("uploads", report.image_name)  # Assuming the highlighted image has the same name

    # Check if the highlighted image exists
    if os.path.exists(highlighted_image_path):
        # Calculate the center position for the image
        image_width = 100  # Width of the image
        image_height = 75  # Height of the image
        x_position = (pdf.w - image_width) / 2  # Center the image horizontally

        # Add the highlighted image
        pdf.image(highlighted_image_path, x=x_position, y=pdf.get_y(), w=image_width, h=image_height)
    else:
        pdf.cell(200, 8, txt="Highlighted image not found.", ln=True)

    # Save the PDF to a temporary file
    pdf_output_path = os.path.join("reports", f"{image_name}.pdf")
    pdf.output(pdf_output_path)

    # Send the PDF file to the client
    return send_file(pdf_output_path, as_attachment=True)

@app.route('/api/detect-forgery', methods=['POST'])
def api_detect_forgery():
    # Check if the file is present in the request
    if 'file' not in request.files:
        print("❌ No file found in request")
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        print("❌ No file selected")
        return jsonify({"success": False, "error": "No selected file"}), 400

    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"✅ File saved at: {filepath}")  # Debug statement
        # ✅ Ensure file is closed before deletion
        # with open(filepath, "rb") as image_file:
        #     image_data = image_file.read()  # Read file to ensure it's not locked

        img = Image.open(filepath)
        print(f"✅ Image received: {filepath}, Size: {img.size}, Format: {img.format}")
        print(f"✅ Image received: {filepath}")
        print(f"   - Size: {img.size}")
        print(f"   - Format: {img.format}")
        print(f"   - Mode: {img.mode}")
        # Check if the model is loaded
        if model is None:
            return jsonify({"success": False, "error": "Model not loaded. Please check the logs."}), 500

        # Perform forgery detection
        class_label, confidence, ela_image = detect_forgery(filepath)
        ela_variance_label, ela_variance_value = calculate_ela_variance(ela_image)
        noise_consistency_label, noise_consistency_value = calculate_noise_consistency(ela_image)
        size, format, mode, file_size = get_image_metadata(filepath)

        # Get additional metadata
        creation_date, modification_date = get_image_creation_date(filepath)
        camera_info = get_camera_info(filepath)
        location_info = get_location_info(filepath)

        # Validate metadata
        validation_results = validate_metadata(creation_date, modification_date, camera_info, location_info)

        # Prepare response data
        response_data = {
            "success": True,
            "result": class_label,
            "confidence": confidence * 100,
            "ela_variance_label": ela_variance_value,
            "noise_consistency_label": noise_consistency_value,
            "metadata": {
                "size": f"{size[0]}x{size[1]}",
                "format": format,
                "mode": mode,
                "file_size": file_size,
                "creation_date": creation_date,
                "modification_date": modification_date,
                "camera_make": camera_info['Make'],
                "camera_model": camera_info['Model'],
                "software": camera_info['Software'],
                "aperture": camera_info['Aperture'],
                "shutter_speed": camera_info['ShutterSpeed'],
                "iso": camera_info['ISO'],
                "focal_length": camera_info['FocalLength'],
                "orientation": camera_info['Orientation'],
                "location": f"{location_info['Latitude']}, {location_info['Longitude']} (Timestamp: {location_info['Timestamp']})"
            },
            "validation_results": {
                "timestamp_mismatch": validation_results['Timestamp Mismatch'],
                "gps_data_valid": validation_results['GPS Data Valid'],
                "editing_software": validation_results['Editing Software']
            }
        }

        return jsonify(response_data)

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        traceback.print_exc()  # ✅ Print full error traceback
        return jsonify({"success": False, "error": str(e)}), 500

    finally:
            time.sleep(1)  # ✅ Give time before deleting
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    print(f"✅ File deleted: {filepath}")
                except Exception as e:
                    print(f"❌ Error deleting file: {e}")


# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)