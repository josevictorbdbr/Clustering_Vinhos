import pickle
import pandas as pd

#abrir dados
dados = pd.read_csv('wine.csv')

#Separar dados vinhos
dados_vinhos = dados.drop(columns=['Wine'], errors='ignore')

#Carregar arquivos após treinamento
preenchedor = pickle.load(open('preenchedor_wine.pkl', 'rb'))
normalizador = pickle.load(open('normalizador_wine.pkl', 'rb'))
cluster_wine = pickle.load(open('cluster_wine.pkl', 'rb'))
colunas_wine = pickle.load(open('colunas_wine.pkl', 'rb'))

#garantir ordem
dados_vinhos = dados_vinhos[colunas_wine]

#preencher valores que faltam
dados_vinhos_preenchidos = preenchedor.transform(dados_vinhos)
dados_vinhos_preenchidos = pd.DataFrame(dados_vinhos_preenchidos, columns=colunas_wine)

#Normalizar
dados_vinhos_norm = normalizador.transform(dados_vinhos_preenchidos)    
dados_vinhos_norm = pd.DataFrame(dados_vinhos_norm, columns=colunas_wine)
    
dados['cluster'] = cluster_wine.predict(dados_vinhos_norm)

#Mostrar quantidade de vinhos por cluster
print('\nQuantidade de vinhos por cluster:\n')
print(dados['cluster'].value_counts().sort_index().to_string())

#Mostrar media das variaveis por cluster
resumo_clusters = dados.drop(columns=['Wine'], errors='ignore').groupby('cluster').mean(numeric_only=True)
print(resumo_clusters)