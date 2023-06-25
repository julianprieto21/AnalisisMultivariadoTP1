import numpy as np
import util

from p01b_logreg import LogisticRegression

def p02cde(train_path, valid_path, test_path, pred_path):
    """Problema 2: regresión logística para positivos incompletos.

    Correr bajo las siguientes condiciones:
        1. en y-labels,
        2. en l-labels,
        3. en l-labels con el factor de correción alfa.

    Args:
        train_path: directorio al CSV conteniendo el archivo de entrenamiento.
        valid_path: directorio al CSV conteniendo el archivo de validación.
        test_path: directorio al CSV conteniendo el archivo de test.
        pred_path: direcotrio para guardar las predicciones.
    """
    pred_path_c = pred_path.replace("WILDCARD", 'c')
    pred_path_d = pred_path.replace("WILDCARD", 'd')
    pred_path_e = pred_path.replace("WILDCARD", 'e')

    # Parte (c): Train y test en labels verdaderos.
    # Asegurarse de guardar las salidas en pred_path_c
    x_train, y_train = util.load_dataset(train_path, label_col="t", add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, label_col="t", add_intercept=True)

    Modelo = LogisticRegression(verbose=True)
    Modelo.fit(x_train, y_train)
    pred = Modelo.predict(x_test)
    np.savetxt(pred_path_c + "/p02c_pred.txt", pred, delimiter=",")

    Modelo.graficos(pred_path_c)
    # pred[pred >= 0.5] = 1 # Redondeo las predicciones
    # pred[pred < 0.5] = 0
    util.plot(x_test, y_test, Modelo.theta, pred_path_c + "/p02c__logreg")  


    # Part (d): Train en y-labels y test en labels verdaderos.
    # Asegurarse de guardar las salidas en pred_path_d
    x_train, y_train = util.load_dataset(train_path, label_col="y", add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, label_col="t", add_intercept=True)

    Modelo = LogisticRegression(verbose=True)
    Modelo.fit(x_train, y_train)
    pred = Modelo.predict(x_test)
    np.savetxt(pred_path_d + "/p02d_pred.txt", pred, delimiter=",")

    Modelo.graficos(pred_path_d)
    pred[pred >= 0.5] = 1 # Redondeo las predicciones
    pred[pred < 0.5] = 0
    util.plot(x_test, y_test, Modelo.theta, pred_path_d + "/p02d_logreg")  

    # Part (e): aplicar el factor de correción usando el conjunto de validación, y test en labels verdaderos.
    # Plot y usar np.savetxt para guardar las salidas en  pred_path_e
    
    x_train, y_train = util.load_dataset(train_path, label_col="y", add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, label_col="t", add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, label_col="y", add_intercept=True)

    Modelo = LogisticRegression(verbose=True)
    Modelo.fit(x_train, y_train)
    valid_pred = Modelo.predict(x_valid) # Predicciones en valid

    valid_positivo = np.sum(y_valid[y_valid==1]) # Cantidad de positivos en valid

    pred_sum = np.sum(valid_pred[y_valid==1]) # Suma de las predicciones positivas en valid

    alpha = pred_sum / valid_positivo # Factor de correción. Formula en el enunciado

    
    Modelo = LogisticRegression(verbose=False)
    Modelo.fit(x_train, y_train, alpha=alpha) # Fit con el factor de correción
    pred = Modelo.predict(x_test) 
    pred = pred / alpha # Aplico el factor de correción a las predicciones

    np.savetxt(pred_path_e + "/p02e_pred.txt", pred, delimiter=",")
    
    Modelo.graficos(pred_path_e)
    # pred[pred >= 0.5] = 1 # Redondeo las predicciones
    # pred[pred < 0.5] = 0
    util.plot(x_test, y_test, Modelo.theta, pred_path_e + "/p02e_logreg", correction=alpha)  
    

# p02cde(
#         train_path="./data/ds3_train.csv",
#         valid_path="./data/ds3_valid.csv",
#         test_path="./data/ds3_test.csv",
#         pred_path=f"output/p02WILDCARD",
#     )