"""Hand Sign Recognition Classifier using MobileNetV2 for heavy datasets"""

import string
from pathlib import Path

import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

english_alphabet = list(string.ascii_uppercase)
numbers = [str(i) for i in range(10)]
classes = english_alphabet + numbers
dict_classes = {label: char for label, char in enumerate(classes)}
num_classes = len(classes)

def create_generators(data_dir:str, batch_size:int=32, img_size:tuple[int, int]=(224, 224), validation_split:float=0.2) -> tuple[ImageDataGenerator, ImageDataGenerator]:
    """Create training and validation generations."""

    # Retrieve images paths
    folders = [f.name for f in Path(data_dir).iterdir() if f.is_dir()]
    print(f"Found {len(folders)} total folders.")
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        validation_split=validation_split
    )
    
    # Validation rescaling
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )
    
    # Create training generator
    train_generator = train_datagen.flow_from_directory(
        directory=data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=folders,
        subset='training',
        shuffle=True,
        seed=42
    )
    
    # Create validation generator
    validation_generator = val_datagen.flow_from_directory(
        directory=data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=folders,
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {validation_generator.samples}")
    return train_generator, validation_generator


def cnn_model(input_shape:tuple[int, int, int], num_classes:int) -> keras.models.Sequential:
    """Build and compile the model."""
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    return model


def main() -> None:
    # Training configuration
    data_dir = "data"
    batch_size = 32
    img_size = (224, 224)
    epochs = 20
    
    # Generators creation
    print("Creating generators...")
    train_gen, val_gen = create_generators(data_dir, batch_size, img_size)
    
    input_shape = (*img_size, 3)
    num_classes = train_gen.num_classes
    print(f"Number of classes: {num_classes}")
    
    # Model building
    print("Building model...")
    model = cnn_model(input_shape, num_classes)
    model.summary()
    
    # Model training
    print("Training...")
    model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // batch_size,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=val_gen.samples // batch_size,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                'data/model.weights.h5',
                save_best_only=True,
                save_weights_only=True,
                monitor='val_accuracy'
            ),
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
        ],
        verbose=1
    )
    
    # Saving final model
    print("Saving final model...")
    model.save('data/model.keras')

if __name__ == "__main__":
    main()
