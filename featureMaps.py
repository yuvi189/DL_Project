# featureMaps.py

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import cv2

# Set up paths
train_malignant_dir = r'C:\Users\sarth\Downloads\DlProject\melanoma_cancer_dataset\train\malignant'
train_benign_dir = r'C:\Users\sarth\Downloads\DlProject\melanoma_cancer_dataset\train\benign'
test_malignant_dir = r'C:\Users\sarth\Downloads\DlProject\melanoma_cancer_dataset\test\malignant'
test_benign_dir = r'C:\Users\sarth\Downloads\DlProject\melanoma_cancer_dataset\test\benign'

def load_and_preprocess_image(img_path, img_size=(224, 224)):
    """Load and preprocess image using MobileNetV2 preprocessing"""
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

def save_model_layer_info(model, filename="model_layer_info.txt"):
    """Save a layer-by-layer breakdown of model components and parameter count to a .txt file"""
    with open(filename, "w") as file:
        file.write("Layer-by-Layer Breakdown of Model\n")
        file.write("="*40 + "\n\n")
        
        for layer in model.layers:
            # Layer details
            layer_name = layer.name
            layer_type = layer.__class__.__name__
            output_shape = layer.output_shape
            num_params = layer.count_params()
            
            # Write layer information
            file.write(f"Layer Name: {layer_name}\n")
            file.write(f"Layer Type: {layer_type}\n")
            file.write(f"Output Shape: {output_shape}\n")
            file.write(f"Number of Parameters: {num_params}\n")
            file.write("-"*40 + "\n")
        
    print(f"Model layer information saved to {filename}")

def visualize_feature_maps(base_model, img_path, output_dir='feature_maps'):
    """Visualize and save feature maps for convolutional layers in the base model"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess image
    img_array = load_and_preprocess_image(img_path)
    
    # Get the original image for display
    orig_img = cv2.imread(img_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    orig_img = cv2.resize(orig_img, (224, 224))
    
    # Save original image
    plt.figure(figsize=(10, 10))
    plt.imshow(orig_img)
    plt.title('Original Image')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'original_image.png'))
    plt.close()

    # Track saved feature maps
    saved_layers = []

    # Create feature map model for each convolutional layer in the base model
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            feature_map_model = Model(inputs=base_model.input, outputs=layer.output)
            
            # Get feature maps
            feature_maps = feature_map_model.predict(img_array)
            
            if feature_maps.ndim == 4:
                n_features = min(16, feature_maps.shape[-1])  # Display up to 16 features
                
                plt.figure(figsize=(20, 10))
                for i in range(n_features):
                    plt.subplot(4, 4, i + 1)
                    feature_map = feature_maps[0, :, :, i]
                    plt.imshow(feature_map, cmap='viridis')
                    plt.axis('off')
                
                plt.suptitle(f'Feature Maps - {layer.name}')
                plt.tight_layout()
                
                # Save feature map
                filename = os.path.join(output_dir, f'feature_maps_{layer.name}.png')
                plt.savefig(filename)
                plt.close()
                
                # Track saved feature maps
                saved_layers.append(layer.name)
                print(f"Saved feature map for layer: {layer.name} at {filename}")

    if not saved_layers:
        print("No feature maps were saved. Check the base model layers for convolutional layers.")

def test_random_images(model, num_images=5):
    """Test model on multiple random images and show predictions"""
    # Collect all image paths
    all_image_paths = []
    for dir_path in [test_malignant_dir, test_benign_dir]:
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_image_paths.append(os.path.join(root, file))
    
    # Randomly select images
    selected_paths = np.random.choice(all_image_paths, num_images, replace=False)
    
    plt.figure(figsize=(20, 4*num_images))
    for idx, img_path in enumerate(selected_paths):
        # Load and preprocess image
        img_array = load_and_preprocess_image(img_path)
        
        # Make prediction
        prediction = model.predict(img_array)[0][0]
        predicted_class = 'Malignant' if prediction > 0.5 else 'Benign'
        actual_class = 'Malignant' if 'malignant' in img_path.lower() else 'Benign'
        
        # Display image and predictions
        plt.subplot(num_images, 2, 2*idx + 1)
        img = image.load_img(img_path, target_size=(224, 224))
        plt.imshow(img)
        plt.title(f'Actual: {actual_class}')
        plt.axis('off')
        
        # Display probability distribution
        plt.subplot(num_images, 2, 2*idx + 2)
        plt.barh(['Benign', 'Malignant'], [1-prediction, prediction])
        plt.xlim(0, 1)
        plt.title(f'Predicted: {predicted_class} ({prediction:.2%})')
    
    plt.tight_layout()
    plt.savefig('prediction_results.png')
    plt.close()
    
    # Also visualize feature maps for one random image
    random_img_path = np.random.choice(selected_paths)
    visualize_feature_maps(model.layers[0], random_img_path)  # Pass base model directly

def main():
    # Load the trained model
    model_path = './melanoma_detection_output/melanoma_detection_model.h5'
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")

        # Save the model layer information breakdown
        save_model_layer_info(model)

        # Test on multiple random images and visualize feature maps for one
        test_random_images(model, num_images=5)
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure that the model path is correct and the model file is compatible.")

if __name__ == "__main__":
    main()
