from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd

# Cargar datos MNIST
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convertir X_train y X_test a DataFrames de pandas
X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

# Entrenar el modelo
model = RandomForestClassifier(n_jobs=-1)
model.fit(X_train, y_train)

# Evaluar el modelo
print(model.score(X_test, y_test))

# Guardar el modelo
with open("model.pkl", "wb") as f:
  joblib.dump(model, f)
