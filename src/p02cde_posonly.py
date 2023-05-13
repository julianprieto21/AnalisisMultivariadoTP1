import numpy as np
import util

from p01b_logreg import LogisticRegression

# Caracter a reemplazar con el sub problema correspondiente.`
WILDCARD = 'c'


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
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** EMPEZAR EL CÓDIGO AQUÍ ***
    # Parte (c): Train y test en labels verdaderos.
    # Asegurarse de guardar las salidas en pred_path_c
    x_train, y_train = util.load_dataset(train_path, label_col="t", add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, label_col="t", add_intercept=True)
    Modelo = LogisticRegression(verbose=False)
    Modelo.fit(x_train, y_train)
    pred = Modelo.predict(x_test)
    np.savetxt(pred_path_c + "/p02c_logreg.csv", pred, delimiter=",")


    # Part (d): Train en y-labels y test en labels verdaderos.
    # Asegurarse de guardar las salidas en pred_path_d
    x_train, y_train = util.load_dataset(train_path, label_col="y", add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, label_col="y", add_intercept=True)
    Modelo = LogisticRegression(verbose=False)
    Modelo.fit(x_train, y_train)
    pred = Modelo.predict(x_test)
    np.savetxt(pred_path_d + "/p02d_logreg.csv", pred, delimiter=",")

    # Part (e): aplicar el factor de correción usando el conjunto de validación, y test en labels verdaderos.
    # Plot y usar np.savetxt para guardar las salidas en  pred_path_e

    # *** TERMINAR CÓDIGO AQUÍ

p02cde(train_path='data/ds3_train.csv', valid_path='data/ds3_valid.csv', test_path='data/ds3_test.csv', pred_path='output/p02{}'.format(WILDCARD))