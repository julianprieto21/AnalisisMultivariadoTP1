from p01b_logreg import p01b
from p01e_gda import p01e
from p02cde_posonly import p02
from p03d_poisson import p03
from p05b_lwr import p05b
from p05c_tau import p05c


correr = X  #CAMBIAR POR número de problema a correr. 0 corre todos.

# Problema 1
if correr == 0 or correr == 1:
    p01b(train_path='../data/ds1_train.csv',
         eval_path='../data/ds1_valid.csv',
         pred_path='output/p01b_pred_1.txt')

    p01b(train_path='../data/ds2_train.csv',
         eval_path='../data/ds2_valid.csv',
         pred_path='output/p01b_pred_2.txt')

    p01e(train_path='../data/ds1_train.csv',
         eval_path='../data/ds1_valid.csv',
         pred_path='output/p01e_pred_1.txt')

    p01e(train_path='../data/ds2_train.csv',
         eval_path='../data/ds2_valid.csv',
         pred_path='output/p01e_pred_2.txt')

# Problema 2
if correr == 0 or correr == 2:
    p02(train_path='../data/ds3_train.csv',
        valid_path='../data/ds3_valid.csv',
        test_path='../data/ds3_test.csv',
        pred_path='output/p02X_pred.txt')

# Problema 3
if correr == 0 or correr == 3:
    p03(lr=1e-7,
        train_path='../data/ds4_train.csv',
        eval_path='../data/ds4_valid.csv',
        pred_path='output/p03d_pred.txt')

# Problema 5
if correr == 0 or correr == 5:
    p05b(tau=5e-1,
         train_path='../data/ds5_train.csv',
         eval_path='../data/ds5_valid.csv')

    p05c(tau_values=[3e-2, 5e-2, 1e-1, 5e-1, 1e0, 1e1],
         train_path='../data/ds5_train.csv',
         valid_path='../data/ds5_valid.csv',
         test_path='../data/ds5_test.csv',
         pred_path='output/p05c_pred.txt')
