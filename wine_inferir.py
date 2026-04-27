import pickle
import pandas as pd

#Carregar arquivos salvos
preenchedor = pickle.load(open('preenchedor_wine.pkl', 'rb'))
normalizador = pickle.load(open('normalizador_wine.pkl', 'rb'))
cluster_wine = pickle.load(open('cluster_wine.pkl', 'rb'))
colunas_wine = pickle.load(open('colunas_wine.pkl', 'rb'))

#Criar novo vinho
novo_vinho = pd.DataFrame([[
    13.20,  #Alcohol
    2.50,   #Malic.acid
    2.40,   #Ash
    18.00,  #Acl
    95.00,  #Mg
    2.00,   #Phenols
    1.80,   #Flavanoids
    0.30,   #Nonflavanoid.phenols
    1.50,   #Proanth
    4.50,   #Color.int
    1.00,   #Hue
    2.50,   #OD
    800.00  #Proline
]], columns=colunas_wine)

#Preencher valores que faltam
novo_vinho_preenchido = preenchedor.transform(novo_vinho)
novo_vinho_preenchido = pd.DataFrame(novo_vinho_preenchido, columns=colunas_wine)

#Normalizar
novo_vinho_norm = normalizador.transform(novo_vinho_preenchido)
novo_vinho_norm = pd.DataFrame(novo_vinho_norm, columns=colunas_wine)

#Cluster do novo vinho
cluster_novo_vinho = cluster_wine.predict(novo_vinho_norm)

print('Cluster do novo vinho:', cluster_novo_vinho[0])