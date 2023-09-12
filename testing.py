import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np




class Testing:
    def __init__(self, model, test_features, test_labels):
        self.model = model
        self.test_features = test_features
        self.test_labels = test_labels
        self.testing()

    def testing(self):
        self.test_predictions = self.model.predict(self.test_features).flatten()
        return
    
    def get_plot(self, history):
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.ylim([0, 3])
        plt.xlabel('Epoch')
        plt.ylabel('Error [TARGET_deathRate]')
        plt.legend()
        plt.grid(True)
        plt.show()
        return
    
    def get_plot_Pred_TrueVal(self):
        a = plt.axes(aspect='equal')
        plt.scatter(self.test_labels, self.test_predictions,c='green')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        lims = [-4, 4]
        plt.xlim(lims)
        plt.ylim(lims)
        _ = plt.plot(lims, lims)
        plt.show()
        return
    


    
    def get_MSE(self):
        Sum = 0
        for i in range(len(self.test_labels)):
            Sum+=(self.test_labels.values[i]-self.test_predictions[i])**2
        res = Sum/len(self.test_labels)
        return res