import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
import torch
import clip
from flask import Flask, request, render_template, jsonify, Response
import io
import requests
import os
from flask_mysqldb import MySQL

app = Flask(__name__)

# MySQL configurations
app.config['MYSQL_HOST'] = 'aws-my-sql-server.c72eswmk0cso.eu-north-1.rds.amazonaws.com'
app.config['MYSQL_USER'] = 'admin'
app.config['MYSQL_PASSWORD'] = 'babusha7038065632'
app.config['MYSQL_DB'] = 'plantClassification'
app.config['MYSQL_PORT'] = 3306  # Default MySQL port

mysql = MySQL(app)

# Load TFLite model
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="resnet_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

# Get input and output details
def get_tflite_io_details(interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return input_details, output_details

# Run inference with TFLite model
def tflite_predict(interpreter, input_data):
    input_details, output_details = get_tflite_io_details(interpreter)
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])
    return output_data

# Load the TFLite model (cached)
tflite_model = load_tflite_model()

# Load CLIP model and preprocessing (cached)
def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

clip_model, clip_preprocess, clip_device = load_clip_model()

# Define class names
class_names = [
    "Nooni", "Nithyapushpa", "Basale", "Pomegranate", "Honge", 
    "Lemon_grass", "Mint", "Betel_Nut", "Nagadali", "Curry_Leaf",
    "Jasmine", "Castor", "Sapota", "Neem", "Ashoka", "Brahmi",
    "Amruta_Balli", "Pappaya", "Pepper", "Wood_sorel", "Gauva",
    "Hibiscus", "Ashwagandha", "Aloevera", "Raktachandini",
    "Insulin", "Bamboo", "Amla", "Arali", "Geranium", "Avacado",
    "Lemon", "Ekka", "Betel", "Henna", "Doddapatre", "Rose",
    "Mango", "Tulasi", "Ganike"
]

# Create a mapping for plant names to their PDF URLs
plant_pdf_map = {
    "Tulasi": "https://kampa.karnataka.gov.in/storage/pdf-files/brochure%20of%20medicinal%20plants/tulsi.pdf",
    "Neem": "https://kampa.karnataka.gov.in/storage/pdf-files/brochure%20of%20medicinal%20plants/neem.pdf",
    "Mint": "https://kampa.karnataka.gov.in/storage/pdf-files/brochure%20of%20medicinal%20plants/mint.pdf",
    "Aloevera": "https://kampa.karnataka.gov.in/storage/pdf-files/brochure%20of%20medicinal%20plants/aloevera.pdf",
    # Add more mappings as needed
}



# Improved validation function using CLIP
def validate_plant_image(image_pil):
    clip_image = clip_preprocess(image_pil).unsqueeze(0).to(clip_device)
    plant_prompts = [
        "a photo of a plant", "a photo of a leaf", "a photo of a green plant",
        "a photo of a medicinal plant", "a photo of herbs"
    ]
    non_plant_prompts = [
        "a photo of a person", "a photo of furniture", "a photo of a building",
        "a photo of a vehicle", "a photo of food", "a random image",
        "a screenshot", "a document scan"
    ]
    all_prompts = plant_prompts + non_plant_prompts
    text_tokens = clip.tokenize(all_prompts).to(clip_device)

    with torch.no_grad():
        logits_per_image, _ = clip_model(clip_image, text_tokens)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    plant_confidence = np.mean(probs[:len(plant_prompts)])
    non_plant_confidence = np.mean(probs[len(plant_prompts):])
    confidence_diff = plant_confidence - non_plant_confidence
    confidence_ratio = plant_confidence / (non_plant_confidence + 1e-6)

    confidence_info = {
        "plant_confidence": round(plant_confidence, 4),
        "non_plant_confidence": round(non_plant_confidence, 4),
        "confidence_diff": round(confidence_diff, 4),
        "confidence_ratio": round(confidence_ratio, 4),
    }

    prompt_scores = {all_prompts[i]: round(probs[i], 4) for i in range(len(all_prompts))}
    is_valid = confidence_diff > 0.1 and confidence_ratio > 1.2

    return is_valid, confidence_info, prompt_scores

# Basic image quality check
def check_image_quality(image_pil):
    img_array = np.array(image_pil)
    brightness = np.mean(img_array)
    too_dark = brightness < 30
    too_bright = brightness > 220
    color_std = np.std(img_array)
    low_contrast = color_std < 15
    too_small = image_pil.width < 100 or image_pil.height < 100
    is_good_quality = not (too_dark or too_bright or low_contrast or too_small)

    quality_info = {
        "brightness": round(float(brightness), 2),
        "color_std": round(float(color_std), 2),
        "size": f"{image_pil.width}x{image_pil.height}",
        "issues": [],
    }

    if too_dark:
        quality_info["issues"].append("Image too dark")
    if too_bright:
        quality_info["issues"].append("Image too bright")
    if low_contrast:
        quality_info["issues"].append("Image has low contrast")
    if too_small:
        quality_info["issues"].append("Image too small")

    return is_good_quality, quality_info


@app.route('/getpdf', methods=['GET'])
def getpdf():
    plant_name = request.args.get('plantName')  # Changed from params to args
    if not plant_name:
        return jsonify({'error': 'Plant name not provided'}), 400
        
    cur = mysql.connection.cursor()
    try:
        cur.execute("SELECT url FROM plantpdf WHERE plant_name = %s", (plant_name,))
        result = cur.fetchone()
        cur.close()
        
        if result:
            return jsonify({'url': result[0]})
        else:
            return jsonify({'error': 'Plant not found'}), 404
    except Exception as e:
        cur.close()
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/proxy-pdf', methods=['GET'])
def proxy_pdf():
    """
    Proxy external PDF URLs through our server to bypass CORS restrictions
    """
    pdf_url = request.args.get('url')
    if not pdf_url:
        return jsonify({'error': 'No URL provided'}), 400
    
    try:
        # Request the PDF with a timeout
        response = requests.get(pdf_url, timeout=10, stream=True)
        
        if response.status_code != 200:
            return jsonify({'error': f'Failed to fetch PDF: HTTP {response.status_code}'}), 400
        
        # Stream the response to avoid loading large PDFs into memory
        return Response(
            response.iter_content(chunk_size=1024),
            content_type=response.headers.get('content-type', 'application/pdf'),
            headers={
                'Content-Disposition': 'inline; filename="plant.pdf"'
            }
        )
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Error fetching PDF: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        image_pil = Image.open(io.BytesIO(file.read()))
        
        # Check image quality
        quality_check, quality_info = check_image_quality(image_pil)
        if not quality_check:
            # Convert numpy values to native Python types for JSON serialization
            quality_info_serializable = {
                'brightness': float(quality_info['brightness']),
                'color_std': float(quality_info['color_std']),
                'size': quality_info['size'],
                'issues': quality_info['issues']
            }
            return jsonify({
                'error': 'Image quality issues',
                'details': quality_info_serializable
            }), 400

        # Validate plant image
        is_valid_plant, confidence_info, _ = validate_plant_image(image_pil)
        if not is_valid_plant:
            # Convert numpy float32 to native Python float
            confidence_info_serializable = {
                'plant_confidence': float(confidence_info['plant_confidence']),
                'non_plant_confidence': float(confidence_info['non_plant_confidence']),
                'confidence_diff': float(confidence_info['confidence_diff']),
                'confidence_ratio': float(confidence_info['confidence_ratio'])
            }
            return jsonify({
                'error': 'Invalid plant image',
                'details': confidence_info_serializable
            }), 400

        # Process image for prediction
        image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (224, 224))
        image = image / 255.0
        image = image.astype(np.float32)
        image = np.expand_dims(image, axis=0)

        # Make prediction
        predictions = tflite_predict(tflite_model, image)
        predicted_index = np.argmax(predictions[0])
        predicted_class_name = class_names[predicted_index]

        # Get top 3 predictions (convert numpy floats to Python floats)
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        top_classes = [
            {"class": class_names[i], "confidence": float(predictions[0][i])} 
            for i in top_indices
        ]

        # Get PDF URL for the predicted plant
        pdf_url = plant_pdf_map.get(predicted_class_name, plant_pdf_map.get("Tulasi"))

        return jsonify({
            'prediction': predicted_class_name,
            'confidence': float(predictions[0][predicted_index]),
            'top_predictions': top_classes,
            'pdf_url': pdf_url
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
