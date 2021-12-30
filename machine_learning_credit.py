import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

with open('data/train_test/credit.pkl', 'rb') as f:
  X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(f)

print(f"{X_credit_treinamento.shape, y_credit_treinamento.shape}")
print(f"{X_credit_teste.shape, X_credit_teste.shape}")

# *************************** NAIVE BAYES *************************************
naive_credit_data = GaussianNB()
naive_credit_data.fit(X_credit_treinamento, y_credit_treinamento)

# Realização da previsão
naive_previsoes = naive_credit_data.predict(X_credit_teste)

# Resultados
print('------------------------------------------------------------')
print('************** ACCURACY SCORE [NAIVE BAYES] ****************')
print(f"{accuracy_score(y_credit_teste, naive_previsoes)}")
print('************** CONFUSION MATRIX [NAIVE BAYES] **************')
print(f"{confusion_matrix(y_credit_teste, naive_previsoes)}")
print('************** ACCURACY SCORE [NAIVE BAYES] ****************')
print(f"{classification_report(y_credit_teste, naive_previsoes)}")

# *************************** DECISION TREE *************************************
arvore_credit = DecisionTreeClassifier(criterion='entropy', random_state = 0)
arvore_credit.fit(X_credit_treinamento, y_credit_treinamento)

# Realização da previsão
decision_previsoes = arvore_credit.predict(X_credit_teste)

# Resultados
print('------------------------------------------------------------')
print('************** ACCURACY SCORE [DECISION TREE] ****************')
print(f"{accuracy_score(y_credit_teste, decision_previsoes)}")
print('************** CONFUSION MATRIX [DECISION TREE] **************')
print(f"{confusion_matrix(y_credit_teste, decision_previsoes)}")
print('************** ACCURACY SCORE [DECISION TREE] ****************')
print(f"{classification_report(y_credit_teste, decision_previsoes)}")

# *************************** RANDOM FOREST *************************************
random_forest_credit = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state = 0)
random_forest_credit.fit(X_credit_treinamento, y_credit_treinamento)

# Realização da previsão
random_previsoes = random_forest_credit.predict(X_credit_teste)

# Resultados
print('------------------------------------------------------------')
print('************** ACCURACY SCORE [RANDOM FOREST] ****************')
print(f"{accuracy_score(y_credit_teste, random_previsoes)}")
print('************** CONFUSION MATRIX [RANDOM FOREST] **************')
print(f"{confusion_matrix(y_credit_teste, random_previsoes)}")
print('************** ACCURACY SCORE [RANDOM FOREST] ****************')
print(f"{classification_report(y_credit_teste, random_previsoes)}")
