import numpy as np
import util
from linear_model import LinearModel
from sklearn.metrics import accuracy_score

def p01e(train_path, eval_path, pred_path, transform=None):
    """Problema 1(e): análisis de discriminante gaussiano (GDA)

    Args:
        train_path: directorio al CSV conteniendo el archivo de entrenamiento.
        eval_path: directorio al CSV conteniendo el archivo de evaluación.
        pred_path: directorio para guardar las predicciones.
    """
    # Se cargan los datos de entrenamiento y de testeo
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=False)

    # Se entrena el modelo y se guardan las predicciones
    Modelo=GDA()
    Modelo.fit(x_train,y_train, transform=transform)
    pred=Modelo.predict(x_valid)
    


    if transform:
        np.savetxt(pred_path + "\p01e_gda_transform.txt", pred,delimiter=',')
        if pred_path[-1] == "1":
            util.plot(x_train, y_train, Modelo.theta, "output/p01h/plot_gda_ds1" + "_transform")
        elif pred_path[-1] == "2":
            util.plot(x_train, y_train, Modelo.theta, "output/p01h/plot_gda_ds2" + "_transform")
    else:
        np.savetxt(pred_path + "\p01e_gda.txt", pred,delimiter=',')
        if pred_path[-1] == "1":
            util.plot(x_train, y_train, Modelo.theta, "output/p01f/plot_gda")
        elif pred_path[-1] == "2":
            util.plot(x_train, y_train, Modelo.theta, "output/p01g/plot_gda")



class GDA(LinearModel):
    """Análisis de discriminante gaussiano.

    Ejemplo de uso:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def normalizar(self, x):
        media = np.mean(x, axis=0) # media de x
        desv = np.std(x, axis=0) # desviacion estandar de x
        x_norm = (x - media) / desv # normaliza x
        print(x_norm)
        return x_norm

    def reciprocidad(self, x):
        return -1/x

    def fit(self, x, y, transform=None):
        """Entrena un modelo GDA.

        Args:
            x: ejemplos de entrenamiento (features solamente). Tamaño (m, n).
            y: etiquetas de ejemplos de entrenamiento. Tamaño (m,).

        Returns:
            theta: parámetros del modelo GDA.
        """

        if transform: x = self.reciprocidad(x) # transforma x si transform es True


        x = np.sqrt(np.abs(x))

        m, n = x.shape # dimensiones de x

        cant_y_1 = y.tolist().count(1) # cantidad de registros clasificados con 1
        cant_y_0 = y.tolist().count(0) # cantidad de registros clasificados con 0
        # print(ind_y_1, ind_y_0)

        sum_x_1 = np.sum(x[y==1],axis=0) # suma los registros clasificados con 1
        sum_x_0 = np.sum(x[y==0],axis=0) # suma los registros clasificados con 0
        # print(sum_x_1, sum_x_0)

        # calcula los parámetros del modelo
        phi = cant_y_1 / m # probabilidad de que 'y' sea 1
        mu_0 = sum_x_0 / cant_y_0 # media de los registros clasificados con 0
        mu_1 = sum_x_1 / cant_y_1 # media de los registros clasificados con 1
        sigma = np.diag((1/m) * np.sum((x - mu_1)**2,axis=0)) # matriz de co-varianza
        # print(phi, mu_0, mu_1, sigma)

        sigma_inv = np.linalg.inv(sigma) # inversa de sigma
        theta = sigma_inv @ (mu_1-mu_0) # calcula theta
        theta_0 = 1/2 * (mu_0.T @ sigma_inv @ mu_0 - mu_1.T @ sigma_inv @ mu_1) - np.log((1-phi)/phi) # calcula theta_0
        self.theta = np.hstack([theta,theta_0]) # concatena theta y theta_0
        # print(self.theta)

        return self.theta


    def predict(self, x):
        """Hace una predicción sobre x nuevos.

        Args:
            x: entradas de tamaño (m, n).

        Returns:
            Salidas de tamaño (m,).
        """
        
        sigmoide = lambda z: 1/(1+np.exp(-z)) # función sigmoide

        theta = self.theta[:2] # parámetros de theta
        theta_0 = self.theta[2] # parámetro de theta_0
        z = (theta @ x.T) + theta_0 # calcula z
        pred = sigmoide(z) # predicciones
        
        return pred
