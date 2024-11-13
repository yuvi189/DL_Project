# project.py

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import pathlib
import seaborn as sns

class MelanomaDetectionSystem:
    def __init__(self, train_dir, val_dir, output_dir, img_size=(224, 224), batch_size=32):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.output_dir = pathlib.Path(output_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._prepare_data()
        self._build_model()

    def _prepare_data(self):
        """Prepare the data generators with augmentation for training and validation"""
        train_datagen = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode="nearest"
        )

        val_datagen = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
        )

        self.train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=True
        )

        self.validation_generator = val_datagen.flow_from_directory(
            self.val_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )

    def _build_model(self):
        """Build a model using MobileNetV2 as base with custom top layers"""
        base_model = MobileNetV2(
            input_shape=(*self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze the base model layers
        base_model.trainable = False

        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
        )

        self.model = model

    def calculate_class_weight(self):
        """Calculate class weight for imbalanced dataset"""
        labels = self.train_generator.classes
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(labels),
            y=labels
        )
        return dict(enumerate(class_weights))

    def create_callbacks(self):
        """Set up callbacks for early stopping and learning rate reduction"""
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        return [early_stopping, reduce_lr]

    def train_model(self, epochs=50):
        # First phase: train only the top layers
        print("Phase 1: Training top layers...")
        self.history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.validation_generator,
            callbacks=self.create_callbacks(),
            class_weight=self.calculate_class_weight(),
            verbose=1
        )

        # Second phase: fine-tune the last few layers of the base model
        print("Phase 2: Fine-tuning MobileNetV2 layers...")
        base_model = self.model.layers[0]
        base_model.trainable = True
        
        # Freeze all layers except the last 30
        for layer in base_model.layers[:-30]:
            layer.trainable = False

        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
        )

        self.history = self.model.fit(
            self.train_generator,
            epochs=epochs//2,
            validation_data=self.validation_generator,
            callbacks=self.create_callbacks(),
            class_weight=self.calculate_class_weight(),
            verbose=1
        )

        # Save the model after training
        self.model.save(self.output_dir / 'melanoma_detection_model.h5')

    def plot_training_history(self):
        """Plot training history for accuracy, loss, precision, recall, and AUC"""
        metrics = ['accuracy', 'loss', 'precision', 'recall', 'auc']
        
        for metric in metrics:
            if metric in self.history.history:
                plt.figure(figsize=(10, 6))
                plt.plot(self.history.history[metric], label=f'Training {metric}')
                plt.plot(self.history.history[f'val_{metric}'], label=f'Validation {metric}')
                plt.title(f'{metric.capitalize()} Over Time')
                plt.xlabel('Epoch')
                plt.ylabel(metric.capitalize())
                plt.legend()
                plt.savefig(self.output_dir / f'{metric}_history.png')
                plt.close()

    def evaluate_model(self):
        """Evaluate model performance"""
        test_predictions = self.model.predict(self.validation_generator)
        test_labels = self.validation_generator.classes
        predicted_classes = (test_predictions > 0.5).astype(int)

        conf_matrix = confusion_matrix(test_labels, predicted_classes)
        class_report = classification_report(test_labels, predicted_classes)

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Benign', 'Malignant'],
                    yticklabels=['Benign', 'Malignant'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(self.output_dir / 'confusion_matrix.png')
        plt.close()

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(test_labels, test_predictions)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(self.output_dir / 'roc_curve.png')
        plt.close()

        with open(self.output_dir / 'classification_report.txt', 'w') as f:
            f.write(class_report)

        return conf_matrix, class_report, roc_auc

def main():
    melanoma_detector = MelanomaDetectionSystem(
        train_dir='C:/Users/sarth/Downloads/DlProject/melanoma_cancer_dataset/train',
        val_dir='C:/Users/sarth/Downloads/DlProject/melanoma_cancer_dataset/test',
        output_dir='melanoma_detection_output'
    )
    
    melanoma_detector.train_model(epochs=50)
    melanoma_detector.plot_training_history()
    conf_matrix, class_report, roc_auc = melanoma_detector.evaluate_model()

    print("\nTraining completed! Check the 'melanoma_detection_output' directory for results.")
    print(f"\nModel Performance:\nROC AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(class_report)

if __name__ == "__main__":
    main()