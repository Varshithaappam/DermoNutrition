from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import numpy as np
import pandas as pd
import cv2

app = Flask(__name__)  # Fixed incorrect _name_ to __name__
model = load_model("skin_cnn_model.keras")

# Load Excel data
def load_disease_info_from_excel(file_path):
    df = pd.read_excel(file_path)
    disease_dict = {}

    for _, row in df.iterrows():
        disease = row["Disease Name"].strip()
        disease_info = {
            "Symptoms": row["Symptoms"],
            "Causes": row["Causes"],
            "Foods to Eat": row["Foods to Eat Regularly"],
            "Foods to Avoid": row["Foods to Avoid"]
        }
        disease_dict[disease] = disease_info

    return disease_dict

disease_dict = load_disease_info_from_excel("labels.xlsx")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def is_skin(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_pixels = np.sum(mask > 0)
    skin_percent = skin_pixels / (img.shape[0] * img.shape[1]) * 100
    return skin_percent > 5

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file or not allowed_file(file.filename):
        return render_template('error.html', error='Only image files (png, jpg, jpeg) are allowed.')

    # Open image and convert to RGB numpy array for skin detection
    image = Image.open(file).convert('RGB')
    img_np = np.array(image)

    if not is_skin(img_np):
        return render_template('error.html', error='The image doesn’t seem to contain skin. Please upload a clear skin image.')

    # Preprocess image for model prediction
    image = image.resize((128, 128))
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    confidence = prediction[0][class_idx]

    classes = [
        "Acne and Rosacea",
        "Actinic Keratosis Basal Cell Carcinoma",
        "Atopic Dermatitis",
        "Cellulitis Impetigo and other Bacterial Infections",
        "Eczema",
        "Exanthems and Drug Eruptions",
        "Herpes HPV and other STDs",
        "Light Diseases and Disorders of Pigmentation",
        "Lupus and other Connective Tissue Diseases",
        "Melanoma Skin Cancer Nevi and Moles",
        "Poison Ivy and other Contact Dermatitis",
        "Psoriasis Lichen Planus and Related Diseases",
        "Seborrheic Keratoses and other Benign Tumors",
        "Systemic Disease",
        "Tinea Ringworm Candidiasis and other Fungal Infections",
        "Urticaria Hives",
        "Vascular Tumors",
        "Vasculitis",
        "Warts Molluscum and other Viral Infections"
    ]

    predicted_class = classes[class_idx]

    if confidence < 0.1:
        return render_template('error.html', error='The result is not conclusive. Please consult a doctor for accurate diagnosis.')

    disease_info = disease_dict.get(predicted_class, {
        "Symptoms": "N/A",
        "Causes": "N/A",
        "Foods to Eat": "N/A",
        "Foods to Avoid": "N/A"
    })

    return render_template(
        'results.html',
        prediction=predicted_class,
        probability=round(confidence * 100, 2),
        symptoms=disease_info["Symptoms"],
        causes=disease_info["Causes"],
        foods_eat=disease_info["Foods to Eat"],
        foods_avoid=disease_info["Foods to Avoid"]
    )

if __name__ == '__main__':  # Fixed incorrect _name_ and _main_
    app.run(debug=True)
