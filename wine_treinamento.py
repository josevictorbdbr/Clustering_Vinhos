import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from scipy.spatial.distance import cdist
import pickle
import math
import numpy as np

#Abrir o arquivo
dados = pd.read_csv('wine.csv')

##Separar valores e dados dos vinhos
dados_vinhos = dados.drop(columns=['Wine'], errors='ignore')

#Separar colunas em categoricas ou numericas
colunas = list(dados_vinhos.columns)

#Tratar valores 
preenchedor= SimpleImputer(strategy='median')
dados_num_preenchidos = preenchedor.fit_transform(dados_vinhos[colunas])
dados_num_preenchidos = pd.DataFrame(dados_num_preenchidos, columns=colunas)

#Salvar Preenchedores
pickle.dump(preenchedor, open('preenchedor_wine.pkl', 'wb'))

#normalizar dados
normalizador = MinMaxScaler()
dados_vinhos_norm = normalizador.fit_transform(dados_num_preenchidos)
dados_vinhos_norm = pd.DataFrame(dados_vinhos_norm, columns=colunas)

print(dados_vinhos_norm)

#salvar normalizador
pickle.dump(normalizador, open('normalizador_wine.pkl', 'wb'))

#distorcoes
distorcoes = []
K = range(1, 11) 

for i in K:
    modelo_clusters = KMeans(n_clusters=i, random_state=42, n_init=10).fit(dados_vinhos_norm)
    distorcoes.append(
        sum(
            np.min(
                cdist(dados_vinhos_norm, modelo_clusters.cluster_centers_,'euclidean'), axis=1
                )/dados_vinhos_norm.shape[0]
            )
        )   
    
#Determinar numero otimo de clusters
x0 = K[0]
y0 = distorcoes[0]
xn = K[-1]
yn = distorcoes[-1]
distancias = []

for i in range(len(distorcoes)):
    x= K[i]
    y= distorcoes[i]
    numerador = abs((yn-y0)*x - (xn-x0)*y + xn*y0 - yn*x0)
    denominador = math.sqrt((yn-y0)**2 + (xn-x0)**2)
    distancias.append(numerador/denominador)
    
numero_clusters_otimo = K[distancias.index(np.max(distancias))]
print('Numero otimo de clusters =', numero_clusters_otimo)

#Treinar modelo final
cluster_wine  = KMeans(n_clusters=numero_clusters_otimo, random_state=42, n_init=10).fit(dados_vinhos_norm)

#salvar modelo
pickle.dump(cluster_wine, open('cluster_wine.pkl', 'wb'))

#salvar nomes colunas
pickle.dump(colunas, open('colunas_wine.pkl', 'wb'))

print('Treinamento concluido')