# Librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn import metrics
from p01b_logreg import LogisticRegression

def show_example(df_x, df_y):
    i = random.randint(0, len(df_x))
    plt.imshow(np.array(df_x.iloc[i, :]).reshape(28,28),cmap="gray")
    plt.colorbar()
    plt.title("Label: " + str(df_y.iloc[i]))
    plt.show()
    

# *** EMPEZAR CÓDIGO AQUÍ ***

# Regresión logística con descenso por gradiente
def p05a(lr, eps, max_iter, train_path, eval_path, pred_path, seed=42):
    # Datos
    cols = pd.read_csv(train_path, nrows=1).columns # columnas
    X = pd.read_csv(train_path, usecols=cols[1:]) # features
    df_y = pd.read_csv(train_path, usecols=["label"]) # labels

    y = df_y["label"] 

    # Imágenes que contengan 3 quedan con label 1 y el resto 0
    
    # show_example(X, y) # mostrar ejemplo

    for i in range(len(y)):
        if y[i] != 1:
            y[i] = 0
        else:
            y[i] = 1


    # Equilibrar cantidad de labels para optimizar el clasificador

    # Seleccionar 1000 imágenes con label 0
    df_0 = X[y == 0] # seleccionar filas con label 0
    # print(df_0.shape)
    df_0 = df_0.sample(n=5000, random_state=seed) # seleccionar 1000 filas (clasificadas con 0) aleatorias

    # Seleccionar 1000 imágenes con label 1
    df_1 = X[y == 1] # seleccionar filas con label 1
    # print(df_1.shape)
    df_1 = df_1.sample(n=5000, random_state=seed) # seleccionar 1000 filas (clasificadas con 1) aleatorias

    # Unir ambos dataframes
    X = pd.concat([df_0, df_1]) # unir dataframes

    df_0_y = y[y == 0].sample(n=5000, random_state=seed) # seleccionar labels con 0
    df_1_y = y[y == 1].sample(n=5000, random_state=seed) # seleccionar labels con 1
    y = pd.concat([df_0_y, df_1_y]) # unir labels

    print(X.shape)
    print(y.shape)
    # show_example(X, y) # mostrar ejemplo

    # Separación en train y test
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42,stratify=y)

    model = LogisticRegression(step_size=lr, eps=eps, max_iter=max_iter, method="gradiente", verbose=True)
    model.fit(x_train, y_train)
    model.graficos(pred_path)
    pred = model.predict(x_test)
    # Guardar predicciones para datos de test (p05_pred1.csv)
    np.savetxt(pred_path + "/pred.csv", pred, delimiter=",")

    # Métricas de evaluación
    pred_copy = pred.copy()
    pred_copy[pred_copy >= 0.5] = 1
    pred_copy[pred_copy < 0.5] = 0
    print(metrics.classification_report(y_test, pred_copy))


# *** TERMINAR CÓDIGO AQUÍ ***


# Entrenar 10 modelos distintos (1 para cada número) con los datos de entrenamiento y
# predecir con ellos la etiqueta de cada imágen en test.
# Guardar las predicciones (p05_predtot.csv). En forma de un vector de 10 elementos
# *** EMPEZAR CÓDIGO AQUÍ ***

def p05b(lr, eps, max_iter, train_path, eval_path, pred_path, seed=42):
    cols = pd.read_csv(train_path, nrows=1).columns # columnas
    X_train = pd.read_csv(train_path, usecols=cols[1:]) # features
    y_train = pd.read_csv(train_path, usecols=["label"])["label"] # labels
    
    # Creacion de modelos
    models = []
    for num in range(10):
        y_copy = y_train.copy()
        for i in range(len(y_copy)):
            if y_copy[i] != num:
                y_copy[i] = 0
            else:
                y_copy[i] = 1
        model = LogisticRegression(step_size=lr, eps=eps, max_iter=max_iter, method="gradiente", verbose=True)
        print("Entrenando modelo para el número " + str(num))
        model.fit(X_train, y_copy)
        models.append(model)
     
    cols = pd.read_csv(eval_path, nrows=1).columns
    X_test = pd.read_csv(eval_path, usecols=cols[1:])
    y_test = pd.read_csv(eval_path, usecols=["label"])["label"]

    # Predicciones
    pred = []
    for i in range(10):
        pred.append(models[i].predict(X_test))
        np.concatenate(pred)
    pred_final = np.array(pred).T
    np.savetxt(pred_path + "/predtot.txt", pred_final, delimiter=",")

    pred_final = np.argmax(pred_final, axis=1)
    # En caso que así se quiera, se pueden visualizar los resultados en una matriz de confusión.
    # Descomentar en caso afirmativo. Siendo y las labels y pred_final la predicción final.

    confusion_matrix = metrics.confusion_matrix(y_test, pred_final)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
    cm_display.plot()
    plt.show()



# *** TERMINAR CÓDIGO AQUÍ ***





# p05a(
#     lr=5e-06, 
#     eps=1e-5, 
#     max_iter=1000, 
#     train_path="./data/mnist_train.csv", 
#     eval_path="./data/mnist_test.csv", 
#     pred_path="output/p05a"
# )

p05b(
    lr=5e-06,
    eps=1e-5,
    max_iter=1000,
    train_path="./data/mnist_train.csv",
    eval_path="./data/mnist_test.csv",
    pred_path="output/p05b"
)
