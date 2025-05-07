import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from src.model import create_model

def train_model(network_input, network_output):
    model = create_model((network_input.shape[1], network_input.shape[2]), len(set(network_output)))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    checkpoint = ModelCheckpoint("models/trained_model.h5", monitor='loss', verbose=1, save_best_only=True, mode='min')
    model.fit(network_input, network_output, epochs=50, batch_size=64, callbacks=[checkpoint])
