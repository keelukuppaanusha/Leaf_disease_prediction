import os
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import cv2
from PIL import Image

# Load pre-trained model
model = load_model('disease_detection_model.h5')

# Class labels for prediction
class_labels = [
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", 
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot", 
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

# Create capture folder if it doesn't exist
capture_folder = "captures"
if not os.path.exists(capture_folder):
    os.makedirs(capture_folder)

# Function to preprocess and predict the disease
def predict_disease(image_path):
    try:
        img = image.load_img(image_path, target_size=(256, 256))  # Ensure resizing to match model input size
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize the image
        
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class = class_labels[predicted_class_index]

        return predicted_class
    except Exception as e:
        return f"Error in prediction: {e}"

# Function to start the webcam for live prediction
esp32_url = "http://192.168.29.79/stream"

def live_prediction():
    cap = cv2.VideoCapture(esp32_url)

    if not cap.isOpened():
        print("Error: Unable to connect to the camera.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow('Live Video Feed', frame)

        # Wait for the user to press 'c' to capture an image and predict
        if cv2.waitKey(1) & 0xFF == ord('c'):
            # Capture and save the image in the captures folder
            img_path = os.path.join(capture_folder, 'live_snapshot.jpg')
            cv2.imwrite(img_path, frame)
            
            # Predict the disease
            prediction = predict_disease(img_path)
            print(f"Predicted Disease: {prediction}")
             # Display the disease name on the frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f"Predicted Disease: {prediction}", (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            if prediction=="Tomato___healthy":
                print("1")
            else:
                print("0")
            
            # Display the disease name on the frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f"Predicted Disease: {prediction}", (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Show the frame with the prediction label
            cv2.imshow('Live Video Feed with Prediction', frame)

            # Show the captured image in the default viewer
            img = Image.open(img_path)
            img = img.resize((250, 250))  # Resize for display
            img.show()  # Use the default image viewer to show the image

            break

    cap.release()
    cv2.destroyAllWindows()

# Start live prediction immediately after running the script
live_prediction()

