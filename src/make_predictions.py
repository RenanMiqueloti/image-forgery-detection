import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def make_predictions(model, val_ds):
    images = []
    true_labels = []
    pred_labels = []
    
    for image_batch, label_batch in val_ds.take(1):
        preds = model.predict(image_batch)
        pred_labels.extend(np.argmax(preds, axis=1))
        true_labels.extend(label_batch.numpy())
        images.extend(image_batch.numpy())
    
    return np.array(images), np.array(true_labels), np.array(pred_labels)

def plot_predictions(images, true_labels, pred_labels):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].astype("uint8"))
        plt.title(f"True: {true_labels[i]}, Pred: {pred_labels[i]}")
        plt.axis("off")
    plt.show()

def plot_history(history):
    plt.figure()
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
