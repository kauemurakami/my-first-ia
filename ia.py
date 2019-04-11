import sklearn
from sklearn import tree

# testes
# print(sklearn.__version__)
# print(tree.__all__)

# Definindo os atributos que você deseja usar para classificar os animais.
features = [[7, 0.6, 40], [7, 0.6, 41], [37, 600, 37], [37, 600, 38]]

#labels = [chicken, chicken, horse, horse]
# we use 0 to represent a chicken and 1 to represent a horse
labels = [0, 0, 1, 1]

# Então definimos o classificador, que será baseado em uma árvore de decisão.
classif = tree.DecisionTreeClassifier()

# Alimente ou ajuste seus dados para o classificador.
classif.fit(features, labels)

# Podemos agora prever resultados a partir de um determinado conjunto de dados. 
# A seguir é mostrado como prever qual animal tem uma altura de 7 polegadas,
# 0,6 kg de peso e uma temperatura de 41 ºC:
print (classif.predict([[7, 0.6, 41]])) #output [0]

# Agora, é mostrado como prever qual animal tem uma altura de 38 polegadas, 
# 600 kg de peso e uma temperatura de 37.5 ºC:
print (classif.predict([[38, 600, 37.5]])) #output [1]
 
# teste proprio 1
print (classif.predict([[35, 500, 38.5]])) #output [1]

# teste proprio 2
print (classif.predict([[3, 1.2, 37]])) #output [0]

