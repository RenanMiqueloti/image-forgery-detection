�
    �ng  �                   �   � d dl mZmZ dd�ZdS )�    )�layers�models��   r   c                 �  � t          j        �   �         }|�                    t          j        | dz   ��  �        �  �         |�                    t          j        ddd��  �        �  �         |�                    t          j        d�  �        �  �         |�                    t          j        ddd��  �        �  �         |�                    t          j        d�  �        �  �         |�                    t          j        d	dd��  �        �  �         |�                    t          j        d�  �        �  �         |�                    t          j        �   �         �  �         |�                    t          j        d	d��  �        �  �         |�                    t          j        d
d��  �        �  �         |�	                    dddg��  �         |S )u  
    Constrói um modelo de rede neural convolucional (CNN) para detecção de falsificação de imagens.

    Args:
        img_size (tuple): Tamanho da imagem de entrada (largura, altura).

    Returns:
        model (tensorflow.keras.Model): Modelo CNN compilado.
    )�   )�input_shape�    )r   r   �relu)�
activation)�   r   �@   r   �   �sigmoid�adam�binary_crossentropy�accuracy)�	optimizer�loss�metrics)
r   �
Sequential�addr   �
InputLayer�Conv2D�MaxPooling2D�Flatten�Dense�compile)�img_size�models     �]/home/renan/Documentos/gitprojects/image-forgery-detection/notebooks/../src/model_training.py�build_modelr"      s�  � � ����E� 
�I�I�f��H�t�O�<�<�<�=�=�=� 
�I�I�f�m�B��6�:�:�:�;�;�;�	�I�I�f�!�&�)�)�*�*�*� 
�I�I�f�m�B��6�:�:�:�;�;�;�	�I�I�f�!�&�)�)�*�*�*� 
�I�I�f�m�C��F�;�;�;�<�<�<�	�I�I�f�!�&�)�)�*�*�*� 
�I�I�f�n������ 
�I�I�f�l�3�6�2�2�2�3�3�3� 
�I�I�f�l�1��3�3�3�4�4�4� 
�M�M�F�)>���M�U�U�U��L�    N)r   )�tensorflow.kerasr   r   r"   � r#   r!   �<module>r&      s;   �� +� +� +� +� +� +� +� +�)� )� )� )� )� )r#   