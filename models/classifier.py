import tensorflow as tf
import numpy as np
import os
import cv2
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2

class DocumentClassifier:
    def __init__(self, model_path=None, categories=None):
        """
        Document classification model using MobileNetV2
        
        Args:
            model_path: Path to pre-trained model
            categories: List of document categories
        """
        self.input_shape = (224, 224, 3)
        self.categories = categories or ['invoice', 'resume', 'receipt']
        self.num_classes = len(self.categories)
        
        if model_path and os.path.exists(model_path):
            self.model = load_model(model_path)
        else:
            self.model = self._build_model()
            
    def _build_model(self):
        """
        Build a CNN model for document classification
        
        Returns:
            Compiled Keras model
        """
        # Use MobileNetV2 as base model
        base_model = MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False
            
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, train_dir, validation_dir=None, epochs=10, batch_size=32):
        """
        Train the model on document images
        
        Args:
            train_dir: Directory with training images
            validation_dir: Directory with validation images
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation
        validation_datagen = ImageDataGenerator(rescale=1./255)
        
        # Flow training images in batches
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        # Flow validation images in batches
        validation_generator = None
        if validation_dir:
            validation_generator = validation_datagen.flow_from_directory(
                validation_dir,
                target_size=self.input_shape[:2],
                batch_size=batch_size,
                class_mode='categorical',
                shuffle=False
            )
            
        # Train the model
        history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size if validation_generator else None
        )
        
        return history
    
    def predict(self, image):
        """
        Predict document class from image
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Tuple of (predicted_class, confidence, all_scores)
        """
        # Preprocess the image
        if isinstance(image, np.ndarray):
            # Resize
            image = cv2.resize(image, self.input_shape[:2])
            
            # Convert to RGB if grayscale
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                
            # Normalize
            image = image.astype(np.float32) / 255.0
        else:
            from PIL import Image
            # Convert PIL Image to numpy array
            image = np.array(image.resize(self.input_shape[:2]))
            
            # Convert to RGB if grayscale
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                
            # Normalize
            image = image.astype(np.float32) / 255.0
            
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        # Predict
        predictions = self.model.predict(image)[0]
        class_idx = np.argmax(predictions)
        confidence = predictions[class_idx]
        predicted_class = self.categories[class_idx]
        
        # Create dictionary of all scores
        all_scores = {self.categories[i]: float(predictions[i]) for i in range(len(self.categories))}
        
        return predicted_class, confidence, all_scores
    
    def save_model(self, model_path):
        """
        Save model to disk
        
        Args:
            model_path: Path to save model
        """
        self.model.save(model_path)
        
    def fine_tune(self, train_dir, validation_dir=None, epochs=5, batch_size=32, learning_rate=1e-4):
        """
        Fine-tune the model by unfreezing some layers
        
        Args:
            train_dir: Directory with training images
            validation_dir: Directory with validation images
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate for fine-tuning
            
        Returns:
            Training history
        """
        # Unfreeze the top layers of the base model
        if isinstance(self.model.layers[0], tf.keras.Model):
            base_model = self.model.layers[0]
            for layer in base_model.layers[-20:]:  # Unfreeze the last 20 layers
                layer.trainable = True
        
        # Recompile the model with a lower learning rate
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train the model
        history = self.train(train_dir, validation_dir, epochs, batch_size)
        
        return history 