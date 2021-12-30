import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

with open('data/train_test/census.pkl', 'rb') as f:
    X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste = pickle.load(f)

print(f"{X_census_treinamento.shape, y_census_treinamento.shape}")
print(f"{X_census_teste.shape, y_census_teste.shape}")
print('============================================================================')

naive_census = GaussianNB()
naive_census.fit(X_census_treinamento, y_census_treinamento)

# Realização da previsão
previsoes = naive_census.predict(X_census_teste)

print('************** ACCURACY SCORE **************')
print(f"{accuracy_score(y_census_teste, previsoes)}")
print('************** CONFUSION MATRIX **************')
print(f"{confusion_matrix(y_census_teste, previsoes)}")
print('************** ACCURACY SCORE **************')
print(f"{classification_report(y_census_teste, previsoes)}")