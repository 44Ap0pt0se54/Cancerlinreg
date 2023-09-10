import pandas as pd
import numpy as np
import model

class DataPreprocess:
    def __init__(self,load):
        if load !='load':
            self.raw_data = None
            self.data = None

            self.load_data(load)
            self.preprocess_data()
            self.save_train_test_data()
        else:
            self.load_data(load)

    def load_data(self,load):
        # Charger les données depuis le fichier CSV (ou autre format de données)
        if load != 'load':
            self.raw_data = pd.read_csv('cancer_reg.csv',encoding='latin-1')
            df_Geo_Pos = pd.read_csv('us-county-boundaries.csv', delimiter=";", encoding='utf-8')[['NAMELSAD','STATE_NAME','INTPTLAT','INTPTLON']]
            self.add_Geo_Pos(df_Geo_Pos) #from Geography(County/State) add Coordinate(x=INTPTLAT,y=INTPTLON)
            self.separate_BinnedInc()

            self.raw_data = self.raw_data.drop('Geography',axis=1)
            self.raw_data = self.raw_data.drop('NAMELSAD',axis=1)
            self.raw_data = self.raw_data.drop('STATE_NAME',axis=1)
            self.raw_data = self.raw_data.drop('PctSomeCol18_24',axis=1) #Too much NaN value
            self.raw_data = self.raw_data.drop('binnedInc',axis=1) 

            self.dicoColumnsNaN = {}
            for index in self.raw_data.index:
                for column in self.raw_data.columns:
                    if   pd.isnull(self.raw_data[column][index]):
                        self.dicoColumnsNaN[column] = self.dicoColumnsNaN.get(column,0)+1
            self.dicoColumnsNaN = dict(sorted(self.dicoColumnsNaN.items()))

            self.data = self.raw_data.copy() # 'PctEmployed16_Over' 
        else :
            self.train_features = pd.read_csv('train_dataset_file')
            self.test_features = pd.read_csv('test_dataset_file')
            self.train_labels = self.train_features.pop('TARGET_deathRate')
            self.test_labels= self.test_features.pop('TARGET_deathRate')
        return
    
        
    def preprocess_data(self):
        # drop NaN Columns
        for key in self.dicoColumnsNaN.keys():
            #self.data = self.data.drop(key,axis=1)
            self.data = self.data_imputation(self.data.copy(),key)

        #Normalization
        self.dataNormalized = (self.data-np.mean(self.data,axis=0))/np.std(self.data)   # Data normalization E = 0, Var = 1

        


        # Diviser les données en ensembles d'entraînement et de test
        self.train_dataset = self.dataNormalized.copy().sample(frac=0.8, random_state=0)
        self.test_dataset = self.dataNormalized.copy().drop(self.train_dataset.index)

        

        self.train_features = self.train_dataset.copy() # Avoid to corrupt data set
        self.test_features = self.test_dataset.copy()


        # Séparer les fonctionnalités (X) de la cible (y)
        self.train_labels = self.train_features.pop('TARGET_deathRate') # Y train
        self.test_labels = self.test_features.pop('TARGET_deathRate') # Y test
        

    def get_train_data(self):
        return self.X_train, self.y_train

    def add_Geo_Pos(self,df_Geo_Pos):
        self.raw_data[['NAMELSAD','STATE_NAME']] = self.raw_data['Geography'].str.extract(r'^(.*?), (.*)$')
        self.raw_data = pd.merge(self.raw_data, df_Geo_Pos, on=['NAMELSAD','STATE_NAME'],how='left')
        return
    
    def separate_BinnedInc(self):
        self.raw_data['binnedInc']=self.raw_data['binnedInc'].str.replace('(','')
        self.raw_data['binnedInc']=self.raw_data['binnedInc'].str.replace('[','')
        self.raw_data['binnedInc']=self.raw_data['binnedInc'].str.replace(']','')
        x=self.raw_data['binnedInc'].str.split(',',expand=True).astype(float)
        x=self.raw_data['binnedInc'].str.split(',',expand=True).astype(float)
        self.raw_data['binnedInc1']=x[0]
        self.raw_data['binnedInc2']=x[1]
        return
    
    def data_normalization(self,data):
        self.m = np.mean(data,axis = 0)
        self.v = np.std(data)
        d = data.copy()

        d.dropna()

        d = (data-self.m)/self.v

        return d
    
    def data_denormalization(self,data_norm):

        d = data_norm.copy()

        d = (data_norm)*self.v+self.m

        return d

    def save_train_test_data(self):
        self.data.to_csv('data_file', index=False)
        self.train_dataset.to_csv('train_dataset_file', index=False)
        self.test_dataset.to_csv('test_dataset_file', index=False)
        return

    def data_imputation(self,train_dataset, column_name): # return train_dataset with predictions instead of NaN for column whose name is column_name
        drop_col = train_dataset.copy()
        ColumnsNaN = drop_col.columns[drop_col.isna().any()].tolist()
        for k in ColumnsNaN:
            drop_col = drop_col.drop(k, axis=1)
        dataWithoutColumnsNaN_normalized = self.data_normalization(drop_col)

        drop_row = train_dataset.copy()
        drop_row = drop_row.dropna()
        dataWithoutRowsNaN_normalized = self.data_normalization(drop_row)

        train_features = dataWithoutRowsNaN_normalized.copy()
        train_labels = train_features.pop(column_name)
        for k in ColumnsNaN:
            if k != column_name:
                train_features = train_features.drop(k, axis=1)

        imputation_model = model.Model.build_and_compile_model(32,16,8,4, len(dataWithoutColumnsNaN_normalized.columns))
        imputation_model.fit(
            train_features,
            train_labels,
            validation_split=0.2,
            verbose=0, epochs=30)

        fill_out = dataWithoutColumnsNaN_normalized.copy()
        fill_out[column_name] = 0

        for i in dataWithoutColumnsNaN_normalized.index:
            if i not in train_features.index:
                fill_out[column_name][i] = imputation_model.predict(dataWithoutColumnsNaN_normalized.loc[[i]])
            else:
                fill_out[column_name][i] = dataWithoutRowsNaN_normalized[column_name][i] 
        fill_out = self.data_denormalization(fill_out)
        for key in self.dicoColumnsNaN:
            if key != column_name:
                fill_out[key]=train_dataset[key]
        return fill_out