�
    Hg�
  �                   �B   � d dl Zd dlmZ d dlZd dlmZ d� Zdd�Zd� Z	dS )�    Nc                 �  � g }g }g }|�                     d�  �        D ]�\  }}| �                    |�  �        }|�                    |dk    �                    t          �  �        �                    �   �         �  �         |�                    |�                    �   �         �  �         |�                    |�                    �   �         �  �         ��t          j        |�  �        t          j        |�  �        t          j        |�  �        fS )u�  
    Faz previsões em um dataset de validação usando o modelo fornecido.

    Args:
        model (tensorflow.keras.Model): Modelo treinado para detecção de falsificação de imagens.
        val_ds (tf.data.Dataset): Dataset de validação contendo imagens e rótulos.

    Returns:
        images (np.array): Array de imagens processadas.
        true_labels (np.array): Rótulos reais.
        pred_labels (np.array): Rótulos previstos pelo modelo.
    �   g      �?)	�take�predict�extend�astype�int�flatten�numpy�np�array)�model�val_ds�images�true_labels�pred_labels�image_batch�label_batch�predss           �R/home/renan/Documentos/gitprojects/image-forgery-detection/src/make_predictions.py�make_predictionsr      s�   � � �F��K��K� %+�K�K��N�N� +� +� ��[����k�*�*�� 	���E�C�K�/�/��4�4�<�<�>�>�?�?�?����;�,�,�.�.�/�/�/����k�'�'�)�)�*�*�*�*��8�F���R�X�k�2�2�B�H�[�4I�4I�I�I�    �   c                 �T  � t          j        d��  �         t          |�  �        D ]o}t          j        d||dz   �  �         t          j        | |         �  �         t          j        d||         � d||         � ��  �         t          j        d�  �         �pt          j        �   �          dS )u+  
    Plota um número selecionado de imagens com suas previsões e rótulos reais.

    Args:
        images (np.array): Array de imagens.
        true_labels (np.array): Rótulos reais.
        pred_labels (np.array): Rótulos previstos.
        num (int): Número de imagens a serem plotadas.
    )�   r   ��figsizer   zTrue: z, Pred: �offN)�plt�figure�range�subplot�imshow�title�axis�show)r   r   r   �num�is        r   �plot_predictionsr)   $   s�   � � �J�w������3�Z�Z� � ����A�s�A��E�"�"�"��
�6�!�9�����	�C�;�q�>�C�C�;�q�>�C�C�D�D�D���������H�J�J�J�J�Jr   c                 �  � | j         d         }| j         d         }| j         d         }| j         d         }t          t          |�  �        �  �        }t          j        d��  �         t          j        ddd�  �         t          j        ||d	�
�  �         t          j        ||d�
�  �         t          j        d��  �         t          j        d�  �         t          j        ddd�  �         t          j        ||d�
�  �         t          j        ||d�
�  �         t          j        d��  �         t          j        d�  �         t          j	        �   �          dS )u�   
    Plota o histórico de treinamento e validação (loss e accuracy).

    Args:
        history (tensorflow.keras.callbacks.History): Histórico de treinamento retornado pelo modelo.
    �accuracy�val_accuracy�loss�val_loss)�   r/   r   r   �   zTraining Accuracy)�labelzValidation Accuracyzlower right)�locz Training and Validation AccuracyzTraining LosszValidation Losszupper rightzTraining and Validation LossN)
�historyr!   �lenr   r    r"   �plot�legendr$   r&   )r3   �acc�val_accr-   r.   �epochs_ranges         r   �plot_historyr:   8   s:  � � �/�*�
%�C��o�n�-�G��?�6�"�D���z�*�H���S���?�?�L� �J�v������K��1�a�����H�\�3�&9�:�:�:�:��H�\�7�*?�@�@�@�@��J�=�!�!�!�!��I�0�1�1�1� �K��1�a�����H�\�4��7�7�7�7��H�\�8�+<�=�=�=�=��J�=�!�!�!�!��I�,�-�-�-��H�J�J�J�J�Jr   )r   )
r   r   �matplotlib.pyplot�pyplotr   �
tensorflow�tfr   r)   r:   � r   r   �<module>r@      s�   �� � � � � � � � � � � � � � � � � � � � �J� J� J�:� � � �(� � � � r   