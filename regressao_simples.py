import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Dados (fáceis!)
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # entrada
y = np.array([2, 4, 6, 8, 10])                # saída

# Criando o modelo
modelo = LinearRegression()

# Treinando
modelo.fit(X, y)

# Fazendo uma previsão
valor = modelo.predict([[6]])

print("Coeficiente:", modelo.coef_[0])
print("Intercepto:", modelo.intercept_)
print("Previsão para X=6:", valor[0])

# Gráfico
plt.scatter(X, y)
plt.plot(X, modelo.predict(X))
plt.title("Regressão Linear Simples")
plt.xlabel("X")
plt.ylabel("y")
plt.show()

