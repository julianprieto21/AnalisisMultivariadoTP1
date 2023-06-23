from p01b_logreg import p01b
from p01e_gda import p01e
from p02cde_posonly import p02cde
from p03d_poisson import p03d
from p05_mnist import p05a, p05b

# from p05b_lwr import p05b
# from p05c_tau import p05c


correr = 2  # CAMBIAR POR número de problema a correr. 0 corre todos.

# Problema 1
if correr == 0 or correr == 1:
    p01b(
        train_path="./data/ds1_train.csv",
        eval_path="./data/ds1_valid.csv",
        pred_path="output/p01b/ds1",
    )

    p01b(
        train_path="./data/ds2_train.csv",
        eval_path="./data/ds2_valid.csv",
        pred_path="output/p01b/ds2",
    )

    p01e(
        train_path="./data/ds1_train.csv",
        eval_path="./data/ds1_valid.csv",
        pred_path="output/p01e/ds1",
        transform=False
    )

    p01e(
        train_path="./data/ds2_train.csv",
        eval_path="./data/ds2_valid.csv",
        pred_path="output/p01e/ds2",
        transform=False,
    )

    #Transformación 1
    p01e(
        train_path="./data/ds1_train.csv",
        eval_path="./data/ds1_valid.csv",
        pred_path="output/p01e/ds1",
        transform=True
    )
    p01e(
        train_path="./data/ds2_train.csv",
        eval_path="./data/ds2_valid.csv",
        pred_path="output/p01e/ds2",
        transform=True
    )

# Problema 2
if correr == 0 or correr == 2:
    p02cde(
        train_path="./data/ds3_train.csv",
        valid_path="./data/ds3_valid.csv",
        test_path="./data/ds3_test.csv",
        pred_path=f"output/p02WILDCARD/p02WILDCARD_pred.txt",
    )

# Problema 3
if correr == 0 or correr == 3:
    p03d(
        lr=1e-7,
        train_path="./data/ds4_train.csv",
        eval_path="./data/ds4_valid.csv",
        pred_path="output/p03d/p03d_pred.txt",
    )

# Problema 5
if correr == 0 or correr == 5:
    p05a(
        lr=5e-06, 
        eps=1e-5, 
        max_iter=1000, 
        train_path="./data/mnist_train.csv", 
        eval_path="./data/mnist_test.csv", 
        pred_path="output/p05a"
    )

    p05b(
        lr=5e-06,
        eps=1e-5,
        max_iter=1000,
        train_path="./data/mnist_train.csv",
        eval_path="./data/mnist_test.csv",
        pred_path="output/p05b"
    )
