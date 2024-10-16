�
    zng�  �                   �   � d dl Zd dlZdd�ZdS )�    N�cifar10��   r   �   c                 ��  �� t          j        | ddgddd��  �        \  \  }}}�fd�}|�                    |t          j        j        j        ��  �        }|�                    |t          j        j        j        ��  �        }|�                    |�  �        �                    t          j        j        j        �  �        }|�                    |�  �        �                    t          j        j        j        �  �        }|||fS )u  
    Carrega e pré-processa os dados do dataset especificado.
    
    Args:
        dataset_name (str): Nome do dataset do TensorFlow Datasets.
        img_size (tuple): Tamanho para redimensionamento das imagens (largura, altura).
        batch_size (int): Tamanho do batch para treinamento.
    
    Returns:
        train_ds (tf.data.Dataset): Dataset de treinamento processado.
        val_ds (tf.data.Dataset): Dataset de validação processado.
        ds_info (tfds.core.DatasetInfo): Informações sobre o dataset.
    �train�testT)�split�shuffle_files�as_supervised�	with_infoc                 �   �� t           j        �                    | ��  �        } t          j        | t           j        �  �        dz  } | |fS )Ng     �o@)�tf�image�resize�cast�float32)r   �label�img_sizes     ��a/home/renan/Documentos/gitprojects/image-forgery-detection/notebooks/../src/data_preprocessing.py�resize_and_normalizez6load_and_preprocess_data.<locals>.resize_and_normalize   s:   �� ������x�0�0�����r�z�*�*�U�2���e�|��    )�num_parallel_calls)	�tfds�load�mapr   �data�experimental�AUTOTUNE�batch�prefetch)�dataset_namer   �
batch_size�train_ds�val_ds�ds_infor   s    `     r   �load_and_preprocess_datar'      s�   �� �  #'�)��������#� #� #���X�v��� � � � � �|�|�0�R�W�EY�Eb�|�c�c�H��Z�Z�,���AU�A^�Z�_�_�F� �~�~�j�)�)�2�2�2�7�3G�3P�Q�Q�H��\�\�*�%�%�.�.�r�w�/C�/L�M�M�F��V�W�$�$r   )r   r   r   )�
tensorflowr   �tensorflow_datasetsr   r'   � r   r   �<module>r+      s;   �� � � � � "� "� "� "�&%� &%� &%� &%� &%� &%r   