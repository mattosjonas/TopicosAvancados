# Jonas Mattos Kunz

# Instalação de Pckages usando referencia https://www.youtube.com/watch?v=8AwF-kOWpMg
# Usando Base de Dados: https://www.kaggle.com/andrewmvd/heart-failure-clinical-data/code
# Local Base de Dados : https://raw.githubusercontent.com/mattosjonas/TopicosAvancados/main/heart_failure_clinical_records_dataset.csv?token=ANYPQJWL65AZSGOVXEV7QQDAYZUF4

print('Randon Forest')

import pandas as pd

bancodeDados = pd.read_csv('https://raw.githubusercontent.com/mattosjonas/TopicosAvancados/main/heart_failure_clinical_records_dataset.csv')
print(bancodeDados.describe())
