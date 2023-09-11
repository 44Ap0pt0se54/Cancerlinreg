import tensorflow as tf
import keras_tuner as kt
import numpy as np
import os
import shutil
import testing

class Model:
    def __init__(self, *arg):
        if type(arg[0]) == int:
            self.layers = arg
            self.model = self.build_and_compile_model(*arg)
            self.save_model()
        elif type(arg[0]) == str:
            self.model = self.load_model(arg[0])

    def build_and_compile_model(self,*layers):   # [l1,l2, ... ,ln, input_size]
        model = tf.keras.Sequential()
        tf.keras.Input(shape=(layers[len(layers)-1],)),
        for elt in layers[0:len(layers)-1]:
            model.add(tf.keras.layers.Dense(elt, activation='relu'))
        model.add(tf.keras.layers.Dense(1))
        model.build((None, layers[len(layers)-1]))
        model.compile(loss='mean_squared_error',
                      optimizer=tf.keras.optimizers.SGD(0.001),
                      metrics=['mse'])
        return model
    
    def build_and_compile_tuned_model(self,hp):   # [l1,l2, ... ,ln, input_size]
        model = tf.keras.Sequential()
        tf.keras.Input(shape=(self.layers[len(self.layers)-1],)),
        for elt in self.layers[0:len(self.layers)-1]:
            model.add(tf.keras.layers.Dense(elt, activation='relu'))
        model.add(tf.keras.layers.Dense(1))
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=hp_learning_rate),
                loss='mse',
                metrics=['mse'])
        return model
    
    def creat_tuned_model(self,train_features,train_labels):
        tuner = kt.Hyperband(self.build_and_compile_tuned_model,
                     objective='mse',
                     max_epochs=10,
                     factor=3)
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        tuner.search(np.array(train_features), np.array(train_labels), epochs=50, validation_split=0.33, callbacks=[stop_early])
        best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
        model = tuner.hypermodel.build(best_hps)
        history = model.fit(np.array(train_features), np.array(train_labels), epochs=50, validation_split=0.33)

        mse_per_epoch = history.history['mse']
        best_epoch = mse_per_epoch.index(min(mse_per_epoch)) + 1
        hypermodel = tuner.hypermodel.build(best_hps)
        hypermodel.fit(np.array(train_features), np.array(train_labels), epochs=best_epoch, validation_split=0.33)
        self.save_model(hypermodel)
        return
    

    
    
    def save_model(self, *arg):
        self.name='ANN-'+str(len(self.layers)-1)+'L'
        for elt in self.layers[0:len(self.layers)-1]:
            self.name = self.name + '-' + str(elt)
        name = self.name
        if os.path.exists('models/'+name):
            shutil.rmtree('models/'+name)
        if len(arg) == 1:
            name += 'HP'
            arg[0].save('models/'+name)
        else:
            self.model.save('models/'+name)
        
        return
    
    def load_model(self,modelPath):
        importedModel = tf.keras.models.load_model('models/'+modelPath)
        return importedModel