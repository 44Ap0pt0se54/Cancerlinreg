import tensorflow as tf
from keras.callbacks import CSVLogger



class Training:
    def __init__(self, model, train_features, train_labels, epochs):
        self.model = model
        self.train_features = train_features
        self.train_labels = train_labels
        self.training(epochs)

    def training(self, epochs):
        self.history = self.model.fit(
            self.train_features,
            self.train_labels,
            epochs=epochs,
            validation_split=0.2,
            verbose=0
        )
        self.train_predictions = self.model.predict(self.train_features).flatten()
        return
    
