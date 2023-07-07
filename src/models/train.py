# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
#import helpers.DataLoader as dtl #Importamos nuestra clase DataLoader com dtl
#from src.helpers.DataLoader import Dataloader as dtl
#from src.helpers import DataLoader as dtl
#from ...src.helpers.DataLoader import DataLoader as dtl
#from  ..helpers.DataLoader import DataLoader as dtl
#@from ..help.DL import Dataloader 
#from src.help.DL import Dataloader as dtl
from src.help import DL as dtl #dlt = DataLoader
# Leer los datos
reader = dtl.Dataloader("./data/processed/RH_procesado.csv")
df = reader.load_data()
# Convertir columnas no numéricas a numéricas
for column in df.columns:
    if df[column].dtype == type(object):
        le = preprocessing.LabelEncoder()
        df[column] = le.fit_transform(df[column])

# Mapear la columna 'Desercion'
mapeo_desercion = {1: 1, 0: 0, 2: 0}
df["Desercion"] = df["Desercion"].replace(mapeo_desercion)

# Suponiendo que 'Desercion' es la columna que quieres predecir
X = df.drop("Desercion", axis=1)
y = df["Desercion"]

# Dividir los datos en conjunto de eWntrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(X_train) #informacion utilizada para entrenar el modelo, el cual es obtenida a traves del archivo csv
print(y_train) #Columna que muestra la prediccion de desercion.

# Entrenar un modelo de Random Forest
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Guardar el modelo entrenado
import joblib

joblib.dump(clf, "./random_forest.joblib")
