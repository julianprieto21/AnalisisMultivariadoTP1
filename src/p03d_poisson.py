import numpy as np
import util

from linear_model import LinearModel


def p03d(lr, train_path, eval_path, pred_path):
    """Problema 3(d): regresión Poisson con ascenso por gradiente.

    Args:
        lr: tasa de aprendizaje para el ascenso por gradiente.
        train_path: directorio al CSV conteniendo el archivo de entrenamiento.
        eval_path: directorio al CSV conteniendo el archivo de evaluación.
        pred_path: direcotrio para guardar las predicciones.
    """
    # Cargar dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** EMPEZAR EL CÓDIGO AQUÍ ***

    # Entrenar una regresión poisson
    # Correr en el conjunto de validación, y usar  np.savetxt para guardar las salidas en pred_path.
    
    # *** TERMINAR CÓDIGO AQUÍ


class PoissonRegression(LinearModel):
    """Regresión poisson.

    Ejemplo de uso:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Corre ascenso por gradiente para maximizar la verosimilitud de una regresión poisson.

        Args:
            x: ejemplos de entrenamiento (features solamente). Tamaño (m, n).
            y: etiquetas de ejemplos de entrenamiento. Tamaño (m,).
        """
        # *** EMPEZAR EL CÓDIGO AQUÍ ***

        # *** TERMINAR CÓDIGO AQUÍ

    def predict(self, x):
        """Hace una predicción sobre x nuevos.

        Args:
            x: entradas de tamaño (m, n).

        Returns:
            Predicción en punto flotante para cada entrada. Tamaño (m,).
        """
        # *** EMPEZAR EL CÓDIGO AQUÍ ***

        # *** TERMINAR CÓDIGO AQUÍ
