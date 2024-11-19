from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import gradio as gr
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# Load YOLO model for object detection
model_yolo = YOLO("./model/classifylogo/best.pt")

# Load the fine-tuned ResNet50 model
model_resnet50 = load_model('./model/fake_new_logo/model_resnet50.h5')

def plot_bboxes(r):
    # Create a copy of the original image to avoid modifying it
    img_copy = np.copy(r.orig_img)
    
    # Initialize Annotator with the copied image
    annotator = Annotator(img_copy)
    
    # Get the bounding boxes from the results
    boxes = r.boxes
    
    # Iterate over each box and draw it on the image
    for box in boxes:
        b = box.xyxy[0]  # Coordinates of the bounding box (xmin, ymin, xmax, ymax)
        c = box.cls  # Class of the box
        annotator.box_label(b, model_yolo.names[int(c)])  # Draw the box with the label
    
    # Return the image with bounding boxes (still in BGR format)
    img = annotator.result() 
    return img


def crop_bboxes(r):
    cropped_images = []  # List to store cropped images
    orig_img = r.orig_img  # Original image
    boxes = r.boxes  # Bounding boxes
    
    for i, box in enumerate(boxes):
        # Get the xyxy coordinates (xmin, ymin, xmax, ymax)
        b = box.xyxy[0].cpu().numpy()  # Convert to CPU before using .numpy()
        xmin, ymin, xmax, ymax = map(int, b)
        
        # Crop the image based on the bounding box coordinates
        cropped_img = orig_img[ymin:ymax, xmin:xmax]
        cropped_images.append(cropped_img)
    
    return cropped_images


def predict_image(model, img):
    # Resize the image to (224, 224)
    img_resized = tf.image.resize(img, (224, 224))  # Resize the image to (224, 224)
    
    # Convert the image to a batch with shape (1, height, width, channels)
    img_array = np.expand_dims(img_resized, axis=0)
    
    # Ensure the array is writable by making a copy of it
    img_array = img_array.copy()
    
    # Preprocess the image for ResNet50
    img_array = preprocess_input(img_array)
    
    # Predict the label
    prediction = model.predict(img_array)
    return prediction


def process_image(input_img):
    # Run YOLO model on the input image
    results = model_yolo(input_img)
    
    r = results[0]

    # Crop bounding boxes and plot the bounding boxes
    cropped_images = crop_bboxes(r)
    img_with_bboxes = plot_bboxes(r)
    
    # Convert BGR to RGB after generating bounding box image
    img_with_bboxes_rgb = cv2.cvtColor(img_with_bboxes, cv2.COLOR_BGR2RGB)
    
    # Convert cropped images from BGR to RGB
    cropped_images_rgb = [cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB) for cropped_img in cropped_images]
    
    # Create a list to store the labels for the cropped images
    labels = []
    
    # Process each cropped image for prediction
    for img in cropped_images:
        prediction = predict_image(model_resnet50, img)
        if prediction[0][0] > 0.5:
            labels.append(f"Genuine with probability {prediction[0][0] * 100:.2f}%")
        else:
            labels.append(f"Fake with probability {(1 - prediction[0][0]) * 100:.2f}%")
    
    # Return the image with bounding boxes in RGB format, cropped images in RGB, and the labels
    return Image.fromarray(img_with_bboxes_rgb), [Image.fromarray(cropped_img) for cropped_img in cropped_images_rgb], labels


# Create the Gradio interface
iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil"),  # Input is an image
    outputs=[gr.Image(type="pil"), gr.Gallery(label="Cropped Images"), gr.Textbox(label="Labels")],  # Outputs: image with boxes, gallery of cropped images, and text labels
    title="Logo Detection with YOLO and ResNet50",
    description="Upload an image to detect logos with YOLO, draw bounding boxes, crop the images, and classify them as Genuine or Fake."
)

# Launch the Gradio interface
iface.launch()
