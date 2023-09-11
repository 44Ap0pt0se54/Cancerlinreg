import tensorflow as tf
from keras.callbacks import CSVLogger



class Training:
    def __init__(self, model, train_features, train_labels, epochs):
        self.model = model
        self.train_features = train_features
        self.train_labels = train_labels
        self.training(epochs)
        self.train_predictions = self.model.predict(self.train_features).flatten()

    def training(self, epochs):
        self.history = self.model.fit(
            self.train_features,
            self.train_labels,
            epochs=epochs,
            validation_split=0.33,
            verbose=0
        )
        return
    
    def get_MSE(self):
        Sum = 0
        for i in range(len(self.train_labels)):
            Sum+=(self.train_labels.values[i]-self.train_predictions[i])**2
        res = Sum/len(self.train_labels)
        return res
    
