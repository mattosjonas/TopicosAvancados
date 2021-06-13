# Jonas Mattos Kunz

# Instalação de Pckages usando referencia https://www.youtube.com/watch?v=8AwF-kOWpMg
# Usando Base de Dados: https://www.kaggle.com/andrewmvd/heart-failure-clinical-data
# Local Base de Dados : https://raw.githubusercontent.com/mattosjonas/TopicosAvancados/main/heart_failure_clinical_records_dataset.csv?token=ANYPQJWL65AZSGOVXEV7QQDAYZUF4

import pandas as pd


print('Algoritmo Naives Bayes')
basedeDados = pd.read_csv('https://raw.githubusercontent.com/mattosjonas/TopicosAvancados/main/heart_failure_clinical_records_dataset.csv')

#print(basedeDados.describe())
#print(basedeDados.columns)

"""
    Necessário definir quais colunas usar
    Todas já padronizadas (Numeros)
    
    Possíveis colunas:
    sex 
    high_bloond_pressure
    diabetes
    smoking
    
    
"""
previsores = basedeDados.iloc[:,1:5].values # observar melhor estas seleções duvidas aqui
classe = basedeDados.iloc[:,4].values # seleciona a coluna high_bloond_pressure

#print(previsores)
#print(classe)

from sklearn.naive_bayes import GaussianNB

classificador = GaussianNB()
classificador.fit(previsores, classe)
print(basedeDados.describe)
print(classificador.classes_)  # possiveis campos
print(classificador.class_count_)
print(classificador.class_prior_)



'''
Copia de codigo do Kaggle para aprendizado Python
import do package import plotly.graph_objs as go
fig = go.Figure()
fig.add_trace(go.Histogram(x=basedeDados['age'],xbins=dict(start = 20,end=95,size=2),))
fig.update_layout(
title_text= 'Age distribution',
xaxis_title_text = 'Idade',
yaxis_title_text= 'Contador',
bargap = 0.05,
template='plotly_dark')
fig.show()

'''




