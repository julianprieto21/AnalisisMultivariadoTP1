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
    np.savetxt(pred_path + "\p01e_gda.txt", pred,delimiter=',')
    # util.plot(x_train, y_train, Modelo.theta, pred_path + "/pred_train.png")
    # util.plot(x_test, y_test, Modelo.theta, pred_path + "/pred_test.png")


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

        m, n = x.shape # dimensiones de x

        ind_y_1 = y.tolist().count(1) # cantidad de registros clasificados con 1
        ind_y_0 = y.tolist().count(0) # cantidad de registros clasificados con 0

        sum_x_1 = np.sum(x[y==1],axis=0) # suma los registros clasificados con 1
        sum_x_0 = np.sum(x[y==0],axis=0) # suma los registros clasificados con 0
        
        # calcula los parámetros del modelo
        phi = ind_y_1/m # probabilidad de que 'y' sea 1
        mu_0 = sum_x_0/ind_y_0 # media de los registros clasificados con 0
        mu_1 = sum_x_1/ind_y_1 # media de los registros clasificados con 1
        sigma = np.diag((1/m)*np.sum((x-mu_1)**2,axis=0)) # varianza de los registros clasificados con 1

        theta = np.linalg.inv(sigma) @ (mu_1-mu_0)
        theta_0 = np.log(phi/(1-phi)) - (1/2)*np.sum((mu_1**2-mu_0**2) @ np.linalg.inv(sigma),axis=0)

        self.theta = np.hstack([theta,theta_0])
        print(self.theta)
        return self.theta

        # *** TERMINAR CÓDIGO AQUÍ ***

    def predict(self, x):
        """Hace una predicción sobre x nuevos.

        Args:
            x: entradas de tamaño (m, n).

        Returns:
            Salidas de tamaño (m,).
        """
        # *** EMPEZAR CÓDIGO AQUÍ ***
        
        sigmoide = lambda z: 1/(1+np.exp(-z)) # función sigmoide
        theta = self.theta[:2]
        theta_0 = self.theta[2]
        z = (theta @ x.T) + theta_0 # calcula z
        pred = sigmoide(z) # predicciones
        return pred

        # *** TERMINAR CÓDIGO AQUÍ ***


