from tensorflow.keras import layers, models, Model
import tensorflow as tf
import numpy as np

def build_model(img_size=(128, 128)):
    inputs = layers.Input(shape=img_size + (3,))
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    print(f"Shape after Flatten: {x.shape}")  # Verificar a forma após Flatten
    x = layers.Dense(512, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    return model

# Construir e testar o modelo
model = build_model()

# Imprimir a arquitetura do modelo
model.summary()

# Testar uma entrada
input_data = np.random.random((1, 128, 128, 3)).astype(np.float32)
output_data = model(input_data)
print(f"Output shape: {output_data.shape}")
