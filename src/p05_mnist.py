# Link a datasets en el pdf de las consignas (analisisMultivariado_tp1_2023cuat1.pdf).
# Dataset utilizado: MNIST (https://drive.google.com/drive/folders/1DAoor-RqKL-zJRNu28jYHoX8QsZZXZxT)

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

    # print(X.shape)
    # print(y.shape)
    # show_example(X, y) # mostrar ejemplo

    # Separación en train y test
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed, stratify=y)



    model = LogisticRegression(step_size=lr, eps=eps, max_iter=max_iter, method="gradiente", verbose=True)
    model.fit(x_train, y_train)
    # model.graficos(pred_path)
    pred = model.predict(x_test)
    # Guardar predicciones para datos de test (p05_pred1.csv)
    np.savetxt(pred_path + "/pred.csv", pred, delimiter=",")

    # Métricas de evaluación
    puntos_cortes = np.arange(0.02, 1, 0.05)
    acc = []
    rec = []
    prec = []
    f1 = []
    for _ in puntos_cortes:
        pred_copy = pred.copy()
        pred_copy[pred_copy >= _] = 1
        pred_copy[pred_copy < _] = 0
        acc.append(metrics.accuracy_score(y_test, pred_copy))
        rec.append(metrics.recall_score(y_test, pred_copy))
        prec.append(metrics.precision_score(y_test, pred_copy))
        f1.append(metrics.f1_score(y_test, pred_copy))
    
    avg_acc = sum(acc) / len(acc)
    avg_recall = sum(rec) / len(rec)
    avg_prec = sum(prec) / len(prec)
    avg_f1 = sum(f1) / len(f1)

    # Graficar métricas
    plt.plot(puntos_cortes, acc, label="Accuracy")
    plt.plot(puntos_cortes, rec, label="Recall")
    plt.plot(puntos_cortes, prec, label="Precision")
    plt.plot(puntos_cortes, f1, label="F1")
    plt.axhline(y=avg_acc, ls="-.", c="blue", label="Promedio Accuracy", alpha=0.7)
    plt.axhline(y=avg_recall, ls="-.", c="orange", label="Promedio Recall", alpha=0.7)
    plt.axhline(y=avg_prec, ls="-.", c="green", label="Promedio Precision", alpha=0.7)
    plt.axhline(y=avg_f1, ls="-.", c="red", label="Promedio F1", alpha=0.7)
    plt.legend()
    plt.show()
    
    pred_copy = pred.copy()
    pred_copy[pred_copy >= 0.65] = 1
    pred_copy[pred_copy < 0.65] = 0
    metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(y_test, pred_copy)).plot()
    plt.show()

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

        # Agregar equilibrio de dataset
    
        model = LogisticRegression(step_size=lr, eps=eps, max_iter=max_iter, method="gradiente", verbose=True)
        print("Entrenando modelo para el número " + str(num))
        model.fit(X_train, y_copy)
        models.append(model)

        # Descomentar para guardar los coeficientes de cada modelo y graficar luego los heatmaps
        # np.savetxt(pred_path + f"/coeff_{num}.csv", model.theta, delimiter=",")
     
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

# p05b(
#     lr=5e-06,
#     eps=1e-5,
#     max_iter=1000,
#     train_path="./data/mnist_train.csv",
#     eval_path="./data/mnist_test.csv",
#     pred_path="output/p05b"
# )

# Hacer mapa de calor 28x28 con los coeficientes de cada modelo
# Ejecutar esto si se guardo los coeficientes de cada modelo en la funcion p05b

# coef_0 = np.loadtxt("./output/p05b/coeficientes/coeff_0.csv", delimiter=",")
# coef_1 = np.loadtxt("./output/p05b/coeficientes/coeff_1.csv", delimiter=",")
# coef_2 = np.loadtxt("./output/p05b/coeficientes/coeff_2.csv", delimiter=",")
# coef_3 = np.loadtxt("./output/p05b/coeficientes/coeff_3.csv", delimiter=",")
# coef_4 = np.loadtxt("./output/p05b/coeficientes/coeff_4.csv", delimiter=",")
# coef_5 = np.loadtxt("./output/p05b/coeficientes/coeff_5.csv", delimiter=",")
# coef_6 = np.loadtxt("./output/p05b/coeficientes/coeff_6.csv", delimiter=",")
# coef_7 = np.loadtxt("./output/p05b/coeficientes/coeff_7.csv", delimiter=",")
# coef_8 = np.loadtxt("./output/p05b/coeficientes/coeff_8.csv", delimiter=",")
# coef_9 = np.loadtxt("./output/p05b/coeficientes/coeff_9.csv", delimiter=",")

# coef_0 = coef_0.reshape(28, 28)
# coef_1 = coef_1.reshape(28, 28)
# coef_2 = coef_2.reshape(28, 28)
# coef_3 = coef_3.reshape(28, 28)
# coef_4 = coef_4.reshape(28, 28)
# coef_5 = coef_5.reshape(28, 28)
# coef_6 = coef_6.reshape(28, 28)
# coef_7 = coef_7.reshape(28, 28)
# coef_8 = coef_8.reshape(28, 28)
# coef_9 = coef_9.reshape(28, 28)

# fig, axs = plt.subplots(2, 5)
# axs[0, 0].imshow(coef_0)
# axs[0, 0].set_title("0")
# axs[0, 1].imshow(coef_1)
# axs[0, 1].set_title("1")
# axs[0, 2].imshow(coef_2)
# axs[0, 2].set_title("2")
# axs[0, 3].imshow(coef_3)
# axs[0, 3].set_title("3")
# axs[0, 4].imshow(coef_4)
# axs[0, 4].set_title("4")
# axs[1, 0].imshow(coef_5)
# axs[1, 0].set_title("5")
# axs[1, 1].imshow(coef_6)
# axs[1, 1].set_title("6")
# axs[1, 2].imshow(coef_7)
# axs[1, 2].set_title("7")
# axs[1, 3].imshow(coef_8)
# axs[1, 3].set_title("8")
# axs[1, 4].imshow(coef_9)
# axs[1, 4].set_title("9")

# for ax in axs.flat:
#     ax.label_outer()
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])

# fig.set_size_inches(10, 5)
# fig.colorbar(axs[0, 0].imshow(coef_0), ax=axs, shrink=1, aspect=10, label="Valor del coeficientes")

# plt.show()





