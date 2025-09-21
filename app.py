import os
import json
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import datetime
import math
import uuid
import io

# Import and configure Cloudinary
import cloudinary
import cloudinary.uploader
import cloudinary.api

# This will be configured by an environment variable on the server
# For local testing, you could temporarily paste your URL here, but do not commit it to git!

cloudinary.config(secure=True)

# --- Initial Setup ---

# NLTK download (Render will run this during the build process)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    print("Downloading VADER lexicon...")
    nltk.download('vader_lexicon')

# Initialize Sentiment Analyzer and Flask App
sia = SentimentIntensityAnalyzer()
app = Flask(__name__)
CORS(app)

# --- Helper Functions ---

def analyze_emotion(text):
    sentiment = sia.polarity_scores(text)
    compound = sentiment['compound']
    if compound > 0.5: return {'emotion': 'Joyful', 'color': '#fde047', 'theme': 'Uplifting'}
    elif compound > 0.05: return {'emotion': 'Positive', 'color': '#86efac', 'theme': 'Peaceful'}
    elif compound < -0.5: return {'emotion': 'Sorrowful', 'color': '#60a5fa', 'theme': 'Somber'}
    elif compound < -0.05: return {'emotion': 'Reflective', 'color': '#c084fc', 'theme': 'Contemplative'}
    else: return {'emotion': 'Neutral', 'color': '#e5e7eb', 'theme': 'Calm'}

def create_and_upload_collage(image_files, memory_id, theme_color):
    if not image_files:
        return None

    images = [Image.open(f).convert("RGBA") for f in image_files]

    num_images = len(images)
    cols = math.ceil(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)
    
    thumb_size = 300
    for i, img in enumerate(images):
        img.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        images[i] = img

    border = 30
    canvas_width = cols * thumb_size + 2 * border
    canvas_height = rows * thumb_size + 2 * border
    collage = Image.new('RGBA', (canvas_width, canvas_height), theme_color)

    x_offset, y_offset = border, border
    for i, img in enumerate(images):
        collage.paste(img, (x_offset, y_offset))
        x_offset += thumb_size
        if (i + 1) % cols == 0:
            x_offset = border
            y_offset += thumb_size
            
    # Save to byte stream
    byte_arr = io.BytesIO()
    collage.save(byte_arr, format='PNG')
    byte_arr.seek(0)

    # Upload the collage to Cloudinary
    collage_public_id = f"memories/{memory_id}/collage"
    upload_result = cloudinary.uploader.upload(
        byte_arr,
        public_id=collage_public_id,
        overwrite=True
    )
    return upload_result['secure_url']


# --- API Endpoints ---

@app.route('/create-memory', methods=['POST'])
def create_memory():
    memory_text = request.form.get('memoryText')
    unlock_date_str = request.form.get('unlockDate')
    files = request.files.getlist('photos')

    if not memory_text or not unlock_date_str:
        return jsonify({'error': 'Missing text or unlock date'}), 400

    memory_id = str(uuid.uuid4())
    
    # 1. Upload original photos to Cloudinary
    image_urls = []
    for file in files:
        if file:
            upload_result = cloudinary.uploader.upload(
                file,
                folder=f"memories/{memory_id}"
            )
            image_urls.append(upload_result['secure_url'])

    # 2. Perform AI processing
    emotion_data = analyze_emotion(memory_text)
    
    # 3. Create and upload the collage
    for file in files:
        file.seek(0)
    collage_url = create_and_upload_collage(files, memory_id, emotion_data['color'])
    
    # 4. Create metadata
    metadata = {
        'id': memory_id,
        'text': memory_text,
        'unlock_date': unlock_date_str,
        'emotion_data': emotion_data,
        'image_urls': image_urls,
        'collage_url': collage_url
    }
    
    # 5. Upload metadata as a JSON file to Cloudinary
    metadata_public_id = f"memories/{memory_id}/metadata"
    metadata_bytes = io.BytesIO(json.dumps(metadata).encode("utf-8"))

    cloudinary.uploader.upload(
        metadata_bytes,
        public_id=metadata_public_id,
        resource_type="raw",  # Important for non-image files
        overwrite=True
    )
        
    return jsonify({'memoryId': memory_id})

@app.route('/get-memory/<memory_id>', methods=['GET'])
def get_memory(memory_id):
    try:
        import requests
        metadata_url = cloudinary.api.resource(f"memories/{memory_id}/metadata", resource_type="raw")['secure_url']
        response = requests.get(metadata_url)
        response.raise_for_status()
        metadata = response.json()
        return jsonify(metadata)
    except Exception as e:
        print(f"Error fetching memory: {e}")
        return jsonify({'error': 'Memory not found or could not be loaded'}), 404    

# --- Page Serving ---

@app.route('/memory/<memory_id>')
def memory_page(memory_id):
    return render_template('memory.html', memory_id=memory_id)

@app.route('/')
def index():
    return render_template('index.html')

# --- Main Execution ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
