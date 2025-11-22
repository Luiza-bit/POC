from sklearn import tree
import matplotlib.pyplot as plt

# -------------------------
# 1. Criando os dados
# -------------------------
# idade, renda (0 = baixa, 1 = média, 2 = alta)
X = [
    [18, 0],  # jovem, renda baixa
    [25, 1],  # adulto, renda média
    [47, 2],  # meia idade, renda alta
    [52, 2],  # adulto, renda alta
    [23, 1],  # jovem, renda média
    [40, 0],  # meia idade, renda baixa
]

# 0 = não compra, 1 = compra
y = [0, 1, 1, 1, 0, 0]

# -------------------------
# 2. Criando o modelo
# -------------------------
modelo = tree.DecisionTreeClassifier()
modelo = modelo.fit(X, y)

# -------------------------
# 3. Fazendo previsão
# -------------------------
# exemplo: pessoa de 30 anos com renda média (1)
previsao = modelo.predict([[30, 1]])
print("Previsão (1 = compra, 0 = não compra):", previsao[0])

# -------------------------
# 4. Mostrar a árvore
# -------------------------
plt.figure(figsize=(12, 8))
tree.plot_tree(modelo, 
               filled=True,
               feature_names=["Idade", "Renda"],
               class_names=["Não compra", "Compra"])
plt.show()
