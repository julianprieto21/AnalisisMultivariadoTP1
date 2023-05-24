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
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)

    # *** EMPEZAR EL CÓDIGO AQUÍ ***

    # Entrenar una regresión poisson
    # Correr en el conjunto de validación, y usar  np.savetxt para guardar las salidas en pred_path.
    Modelo = PoissonRegression(step_size=lr)
    Modelo.fit(x_train, y_train)
    # pred = Modelo.predict(x_eval)
    # np.savetxt(pred_path, pred, delimiter=",")


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
        
        m, n = x.shape

        if self.theta is None:
            self.theta = np.zeros(n)

        gradiente = lambda theta: np.sum(y)/(theta @ x.T) - m

        for i in range(self.max_iter):
            theta = self.theta

            grad = gradiente(theta)
            print(grad)
            self.theta = theta - self.step_size @ grad
        
            error = np.linalg.norm(self.theta - theta)
            if error < self.eps:
                break
        
        print("Iteraciones: ", i)
        print("Error: ", error)
        print("Theta: ", self.theta)

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
