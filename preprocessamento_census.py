import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder # Converte atributos categóricos em numéricos
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split # Divisão entre TREINAMENTO e TESTE
from sklearn.preprocessing import StandardScaler     # Escalonamento dos valores
from imblearn.over_sampling import SMOTE

base_census = pd.read_csv('data/census.csv')

print('=============================================================================')
print(base_census.describe())
print('=============================================================================')
print(base_census.isnull().sum())
print('=============================================================================')
print(base_census.columns)
print('=============================================================================')

# Divisão entre previsores e classe
X_census = base_census.iloc[:, 0:14].values
y_census = base_census.iloc[:, 14].values

print('----------------------------- DE ----------------------------------')
print(X_census[0])

# *************************************************************************************
# Tratamento de atributos categóricos
# *************************************************************************************

'''
 (LabelEncoder) - converte os atributos categóricos em números.
'''
label_encoder_workclass = LabelEncoder()
label_encoder_education = LabelEncoder()
label_encoder_marital = LabelEncoder()
label_encoder_occupation = LabelEncoder()
label_encoder_relationship = LabelEncoder()
label_encoder_race = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_country = LabelEncoder()

X_census[:,1] = label_encoder_workclass.fit_transform(X_census[:,1])
X_census[:,3] = label_encoder_education.fit_transform(X_census[:,3])
X_census[:,5] = label_encoder_marital.fit_transform(X_census[:,5])
X_census[:,6] = label_encoder_occupation.fit_transform(X_census[:,6])
X_census[:,7] = label_encoder_relationship.fit_transform(X_census[:,7])
X_census[:,8] = label_encoder_race.fit_transform(X_census[:,8])
X_census[:,9] = label_encoder_sex.fit_transform(X_census[:,9])
X_census[:,13] = label_encoder_country.fit_transform(X_census[:,13])

print('----------------------------- PARA ----------------------------------')
print(X_census[0])

'''
 (OneHotEncoder) - garante a representatividade dos atributos.

Ex.:
        # Carro

        # Gol Pálio Uno
        #   1     2   3

        # Gol   1 0 0
        # Pálio 0 1 0
        # Uno   0 0 1 # encode
'''
onehotencoder_census = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1,3,5,6,7,8,9,13])], remainder='passthrough')
X_census = onehotencoder_census.fit_transform(X_census).toarray()

print(X_census.shape)
print('============================================================================')

# ************************************ BALANCEAMENTO - técnica SMOTE **************************************
smote = SMOTE(sampling_strategy='minority')
X_census_balance, y_census_balance = smote.fit_resample(X_census, y_census)

# Desbalanceado
print(np.unique(y_census, return_counts=True))           # renda (income) <=50K e >50K 
# Balanceado  
print(np.unique(y_census_balance, return_counts=True))   # renda (income) <=50K e >50K  
print('============================================================================')

# ************************************ ESCALONAMENTO DOS VALORES ******************************************
scaler_census = StandardScaler()
X_census = scaler_census.fit_transform(X_census_balance)

# **************************** DIVISÃO DAS BASES EM TREINAMENTO E TESTE ***********************************
X_census_treinamento, X_census_teste, y_census_treinamento, y_census_teste = train_test_split(X_census_balance, y_census_balance, test_size = 0.15, random_state = 0)

print('BASES DE TREINAMENTO E TESTE')
print(f"{X_census_treinamento.shape, y_census_treinamento.shape}")
print(f"{X_census_teste.shape, y_census_teste.shape}")
print('============================================================================')

# Salvar as variáveis
with open('data/train_test/census.pkl', mode = 'wb') as f:
  pickle.dump([X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste], f)