import numpy as np
import util
from linear_model import LinearModel
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def p01b(train_path, eval_path, pred_path=""):
    """Problema 1(b): Regresión Logística con el método de Newton.

    Args:
        train_path: directorio al CSV conteniendo el archivo de entrenamiento.
        eval_path: directorio al CSV conteniendo el archivo de evaluación.
        pred_path: directorio para guardar las predicciones.
    """
    # Se cargan los datos de entrenamiento y de testeo
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_test, y_test = util.load_dataset(eval_path, add_intercept=True)

    # Se entrena el modelo y se guardan las predicciones
    Modelo = LogisticRegression()
    Modelo.fit(x_train, y_train)
    Modelo.graficos(pred_path)
    pred = Modelo.predict(x_test)
    np.savetxt(pred_path + "/p01b_logreg.csv", pred, delimiter=",")


class LogisticRegression(LinearModel):
    """Regresión Logística con Newton como solver.

    Ejemplo de uso:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def reglog(self, x, coef):
        """Corre una regresión logística para x y devuelve la predicción.

        Args:
            x: Conjunto de datos. Tamaño (m, n).
            coef: Coeficientes de la regresión. Tamaño (m,).

        Returns:
            Salidas de tamaño (m,).
        """

        # *** EMPEZAR CÓDIGO AQUÍ ***
        # *** TERMINAR CÓDIGO AQUÍ ***

    def fit(self, x, y):
        """Corre el método de Newton para minimizar J(tita) para reg log.

        Args:
            x: ejemplos de entrenamiento (features solamente). Tamaño (m, n).
            y: etiquetas de ejemplos de entrenamiento. Tamaño (m,).
        """
        # *** EMPEZAR CÓDIGO AQUÍ ***

        m, n = x.shape
        # inicializa theta en vector de ceros si es None
        if self.theta is None:
            self.theta = np.zeros(n)

        # inicializa la funcion sigmoide, el gradiente y el hessiano
        def sigmoide(theta):
            return 1 / (1 + np.exp(-theta @ x.T))  # @ producto matricial

        def costo(theta):
            valores_sigmoide = sigmoide(theta)
            return (
                -1
                / m
                * (
                    y @ np.log(valores_sigmoide)
                    + (1 - y) @ np.log(1 - valores_sigmoide)
                )
            )

        def gradiente(theta):
            valores_sigmoide = sigmoide(theta)
            return x.T @ (valores_sigmoide - y)

        def hessiano(theta):
            valores_sigmoide = sigmoide(theta)
            diag = np.diag(valores_sigmoide * (1 - valores_sigmoide))
            return x.T @ diag @ x

        # setea el error inicial en infinito para entrar al while
        error = np.Infinity

        # comienza con las iteraciones con el metodo de Newton
        while error > self.eps and self.contador_iteraciones < self.max_iter:
            grad = gradiente(self.theta)
            hess = hessiano(self.theta)
            hess_inv = np.linalg.inv(hess)
            new_theta = self.theta - hess_inv @ grad
            error = np.linalg.norm(new_theta - self.theta)
            self.theta = new_theta
            self.contador_iteraciones += 1

            self.coeficientes.append(self.theta)
            self.costo.append(costo(self.theta))

            pred = sigmoide(self.theta)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            self.accuracy.append(accuracy_score(y, pred))

        # si verbose es True, imprime los resultados
        if self.verbose:
            print("Terminó en", self.contador_iteraciones, "iteraciones")
            print("Error:", error)
            print("Theta:", self.theta)
            # print("Costos", self.costo)
            # print("Coeficientes", self.coeficientes)

        # *** TERMINAR CÓDIGO AQUÍ ***

    def predict(self, x):
        """Hace una predicción sobre x nuevos.

        Args:
            x: entradas de tamaño (m, n).

        Returns:
            Salidas de tamaño (m,).
        """
        # *** EMPEZAR CÓDIGO AQUÍ ***

        def prob_1(theta):
            return 1 / (1 + np.exp(-theta @ x.T))

        def prob_0(theta):
            return 1 - (1 / (1 + np.exp(-theta @ x.T)))

        probs = prob_1(self.theta)
        probs_copy = probs.copy()
        for i in range(len(probs)):
            if probs[i] < 0.75:
                probs_copy[i] = 0
            else:
                probs_copy[i] = 1
        return probs_copy

        # *** TERMINAR CÓDIGO AQUÍ ***

    def graficos(self, pred_path):
        """Crea los siguientes gráficos.

        - Costo vs Iteraciones
        - Accuracy de entrenamiento vs Iteraciones
        - Evolución features (sin graficar el intercept)
        Args:
            pred_path: directorio para guardar las imágenes.
        """
        # *** EMPEZAR CÓDIGO AQUÍ ***

        # Costo vs Iteraciones
        plt.plot(self.costo)
        plt.xlabel("Iteraciones")
        plt.ylabel("Costo")
        plt.title("Costo vs Iteraciones")
        plt.savefig(pred_path + "/p01b_costo_iteraciones.png")

        # Accuracy de entrenamiento vs Iteraciones
        plt.clf()  # limpia el gráfico anterior
        plt.plot(self.accuracy)
        plt.xlabel("Iteraciones")
        plt.ylabel("Accuracy")
        plt.title("Accuracy de entrenamiento vs Iteraciones")
        plt.legend(["Accuracy con corte en 0.5"])
        plt.savefig(pred_path + "/p01b_accuracy_iteraciones.png")

        # Evolución features (sin graficar el intercept)
        plt.clf()  # limpia el gráfico anterior
        for i in range(1, len(self.coeficientes[0])):
            plt.plot([theta[i] for theta in self.coeficientes])
        plt.xlabel("Iteraciones")
        plt.ylabel("Coeficientes")
        plt.title("Evolución features")
        plt.legend(["Feature 1", "Feature 2"])
        plt.savefig(pred_path + "/p01b_evolucion_features.png")

        # *** TERMINAR CÓDIGO AQUÍ ***


p01b("data/ds1_train.csv", "data/ds1_valid.csv", "output")
