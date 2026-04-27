# Wine Clustering

Este projeto usa o algoritmo K-Means para agrupar vinhos em diferentes grupos com base nas características usando de base o dataset abaixo.

**Dataset:** Wine.csv 

https://gist.github.com/tijptjik/9408623

## Como Rodar

1. Dentro da pasta, criar e ativar o ambiente virtual
2. Instalar as dependências dentro do **requirements.txt**
3. Rodar o arquivo **wine_treinamento.py** para ler o dataset, normalizar os dados, treinar o modelo de clusters e salvar os arquivos .pkl

## Módulos

- **wine_descrever.py**  Mostra quantos vinhos existe em cada cluster
- **wine_inferir.py**  Cria um vinho manualmente e informa a qual cluster ele pertence
