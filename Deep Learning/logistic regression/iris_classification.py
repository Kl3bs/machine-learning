import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Carregando a base de dados iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Dividindo a base de dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Padronizando os dados (média zero e variância unitária)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Criando o modelo de regressão logística multinomial
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

# Treinando o modelo
model.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliando a precisão do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy:.2f}')

# Exibindo relatório de classificação
class_report = classification_report(y_test, y_pred, target_names=iris.target_names)
print('Relatório de Classificação:\n', class_report)
