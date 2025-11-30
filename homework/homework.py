# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#



import json
import gzip
import pickle
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

def cargar_csv(ruta):
    return pd.read_csv(ruta, compression="zip")

def limpiar_dataset(df):
    df = df.rename(columns={"default payment next month": "default"})
    df = df.drop(columns=["ID"], errors="ignore")
    df = df.dropna()
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: x if x in [0, 1, 2, 3, 4] else 4)

    return df

def separar_xy(df):
    return df.drop(columns=["default"]), df["default"]


def crear_pipeline():
    categ = ["SEX", "EDUCATION", "MARRIAGE"]

    encoder = ColumnTransformer(
        transformers=[("cat_encoder", OneHotEncoder(handle_unknown="ignore"), categ)],
        remainder="passthrough"
    )

    flujo = Pipeline([
        ("prep", encoder),
        ("rf", RandomForestClassifier(random_state=42))
    ])

    return flujo

def ajustar_hiperparametros(paso, Xtr, Ytr):
    grid = {
        "rf__n_estimators": [50, 100, 200],
        "rf__max_depth": [None, 5, 10, 20],
        "rf__min_samples_split": [2, 5, 10],
        "rf__min_samples_leaf": [1, 2, 4],
    }

    buscador = GridSearchCV(
        paso,
        grid,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        verbose=1
    )

    buscador.fit(Xtr, Ytr)
    return buscador


def guardar_modelo(modelo, ruta):
    carpeta = os.path.dirname(ruta)
    os.makedirs(carpeta, exist_ok=True)
    with gzip.open(ruta, "wb") as f:
        pickle.dump(modelo, f)


def evaluar_metricas(y_real, y_hat, nombre):
    return {
        "type": "metrics",
        "dataset": nombre,
        "precision": precision_score(y_real, y_hat, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_real, y_hat),
        "recall": recall_score(y_real, y_hat, zero_division=0),
        "f1_score": f1_score(y_real, y_hat, zero_division=0),
    }


def matriz_confusion_dict(y_real, y_hat, nombre):
    cm = confusion_matrix(y_real, y_hat)
    return {
        "type": "cm_matrix",
        "dataset": nombre,
        "true_0": {"predicted_0": int(cm[0][0]), "predicted_1": int(cm[0][1])},
        "true_1": {"predicted_0": int(cm[1][0]), "predicted_1": int(cm[1][1])},
    }

def escribir_json(lista, ruta):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "w") as f:
        for item in lista:
            f.write(json.dumps(item) + "\n")


def ejecutar():
    ruta_train = "files/input/train_data.csv.zip"
    ruta_test = "files/input/test_data.csv.zip"
    ruta_modelo = "files/models/model.pkl.gz"
    ruta_metricas = "files/output/metrics.json"

    # Cargar
    df_tr = cargar_csv(ruta_train)
    df_te = cargar_csv(ruta_test)

    # Preprocesar
    df_tr = limpiar_dataset(df_tr)
    df_te = limpiar_dataset(df_te)

    Xtr, Ytr = separar_xy(df_tr)
    Xte, Yte = separar_xy(df_te)

    # Modelo
    modelo = crear_pipeline()
    modelo_opt = ajustar_hiperparametros(modelo, Xtr, Ytr)

    # Guardar modelo
    guardar_modelo(modelo_opt, ruta_modelo)

    # Predicciones
    pred_tr = modelo_opt.predict(Xtr)
    pred_te = modelo_opt.predict(Xte)

    # Métricas
    met_tr = evaluar_metricas(Ytr, pred_tr, "train")
    met_te = evaluar_metricas(Yte, pred_te, "test")

    cm_tr = matriz_confusion_dict(Ytr, pred_tr, "train")
    cm_te = matriz_confusion_dict(Yte, pred_te, "test")

    # Guardar a archivo
    escribir_json([met_tr, met_te, cm_tr, cm_te], ruta_metricas)


if __name__ == "__main__":
    ejecutar()