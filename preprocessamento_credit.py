# Pylance
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split # Divisão entre TREINAMENTO e TESTE
from sklearn.preprocessing import StandardScaler     # Escalonamento dos valores
from imblearn.over_sampling import SMOTE

# Carregamento dos dados
base_credit = pd.read_csv('data/credit_data.csv')

print(base_credit)
print('=============================================================================')
print(base_credit.describe())
print('=============================================================================')
print(np.unique(base_credit['default'], return_counts=True)) # 0 - pagou , 1 - não pagou
print('=============================================================================')

# Tratamento de valores inconsistentes
base_credit.loc[base_credit['age'] < 0, 'age'] = base_credit['age'].mean()

# Tratamento de valores faltantes
base_credit['age'].fillna(base_credit['age'].mean(), inplace = True)

# Divisão entre previsores e classe
X_credit = base_credit.iloc[:, 1:4].values
y_credit = base_credit.iloc[:, 4].values

# Balanceamento - técnica SMOTE
smote = SMOTE(sampling_strategy='minority')
X_credit_balance, y_credit_balance = smote.fit_resample(X_credit, y_credit)

# Desbalanceado
print(np.unique(y_credit, return_counts=True))          # 0 - pagou , 1 - não pagou   
# Balanceado  
print(np.unique(y_credit_balance, return_counts=True))  # 0 - pagou , 1 - não pagou    

print('============================================================================')

# Escalonamento dos valores
scaler_credit = StandardScaler()
X_credit_balance = scaler_credit.fit_transform(X_credit_balance)

# Divisão das base em TREINAMENTO e TESTE
X_credit_treinamento, X_credit_teste, y_credit_treinamento, y_credit_teste = train_test_split(X_credit_balance, y_credit_balance, test_size = 0.25, random_state = 0)
print(f"{X_credit_treinamento.shape, y_credit_treinamento.shape}")
print(f"{X_credit_teste.shape, X_credit_teste.shape}")
print('============================================================================')

# Salvar as variáveis
with open('data/train_test/credit.pkl', mode = 'wb') as f:
  pickle.dump([X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste], f)



