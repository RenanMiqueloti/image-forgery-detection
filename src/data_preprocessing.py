import tensorflow as tf
import tensorflow_datasets as tfds

def load_and_preprocess_data(dataset_name='cifar10', img_size=(128, 128), batch_size=16):  # Reduzido o batch size para 16
    (train_ds, val_ds), ds_info = tfds.load(
        dataset_name,
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True
    )

    def resize_and_normalize(image, label):
        image = tf.image.resize(image, img_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    train_ds = train_ds.map(resize_and_normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.map(resize_and_normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)

    # Adiciona o prefetch para melhorar o desempenho
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.experimental.AUTOTUNE)

    return train_ds, val_ds
