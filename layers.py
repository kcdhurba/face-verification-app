# Custom L1 Distance layer module 

# Import dependencies 
import tensorflow as tf
from tensorflow.keras import Layer

# Custom l1 Distance layer 

class L1Dist(Layer): 
    # init method -- inheritence 
    def __init__(self, **kwargs): 
        super().__init__()

    def call(self, input_embedding, validation_embedding): 
        return tf.math.abs(input_embedding - validation_embedding)
    

