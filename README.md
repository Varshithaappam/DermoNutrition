DermoNutrition

DermoNutrition is a skin disease detection system using a Convolutional Neural Network (CNN) model. After predicting the skin disease from an uploaded image, the app also recommends food to eat and avoid based on the diagnosis.

Features
- Upload an image of a skin condition
- Detects 19 types of skin diseases using a trained CNN model
- Displays:
  - Disease name
  - Symptoms and causes
  - Recommended food
  - Food to avoid

 Files
- main.py – Flask backend for image upload and prediction
- skin_cnn_model.keras – Trained EfficientNet-based model
- labels.xlsx – Contains disease names, symptoms, and food recommendations



 How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
2. Run the app:

python main.py


3. Open http://localhost:5000 in your browser.

Output
![Screenshot 2025-06-13 121133](https://github.com/user-attachments/assets/28647442-3339-4693-b3bf-98fea6dea2ac)
![Screenshot 2025-06-13 121156](https://github.com/user-attachments/assets/ef255825-b88b-4e10-b388-52701af86481)
![Screenshot 2025-06-13 121209](https://github.com/user-attachments/assets/34d0f96f-1051-4478-9755-1bb16dc250be)


