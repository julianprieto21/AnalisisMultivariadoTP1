import numpy as np
import util
from linear_model import LinearModel
from sklearn.metrics import accuracy_score

def p01e(train_path, eval_path, pred_path):
    """Problema 1(e): análisis de discriminante gaussiano (GDA)

    Args:
        train_path: directorio al CSV conteniendo el archivo de entrenamiento.
        eval_path: directorio al CSV conteniendo el archivo de evaluación.
        pred_path: directorio para guardar las predicciones.
    """
    # Se cargan los datos de entrenamiento y de testeo
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_test, y_test = util.load_dataset(eval_path, add_intercept=False)

    # Se entrena el modelo y se guardan las predicciones
    Modelo=GDA()
    Modelo.fit(x_train,y_train)
    pred=Modelo.predict(x_test)
    np.savetxt(pred_path + "\p01e_gda.csv", pred,delimiter=',')


class GDA(LinearModel):
    """Análisis de discriminante gaussiano.

    Ejemplo de uso:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Entrena un modelo GDA.

        Args:
            x: ejemplos de entrenamiento (features solamente). Tamaño (m, n).
            y: etiquetas de ejemplos de entrenamiento. Tamaño (m,).

        Returns:
            theta: parámetros del modelo GDA.
        """
        # *** EMPEZAR CÓDIGO AQUÍ ***
        
        # *** TERMINAR CÓDIGO AQUÍ ***

    def predict(self, x):
        """Hace una predicción sobre x nuevos.

        Args:
            x: entradas de tamaño (m, n).

        Returns:
            Salidas de tamaño (m,).
        """
        # *** EMPEZAR CÓDIGO AQUÍ ***
        
        # *** TERMINAR CÓDIGO AQUÍ ***


