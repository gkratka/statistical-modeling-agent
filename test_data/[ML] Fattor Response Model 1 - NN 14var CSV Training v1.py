# ______________________________________________________________________
# INSTRUCTIONS

# MODELO: propensao a fazer promessa
# PROPOSITO DESSE SCRIPT: treinar a rede neural
# 0. rodadas de treino da rede reural = 8 (modelos com parametros diferentes)
# 1. path0: localizacao da base de treino (csv)
# 2. colocar a variavel resposta (se promessa foi feita) como ultima coluna padronizando valores para 0 ou 1
# 3. numero de variaveis independentes: 14
# 4. variaveis independentes: ID_CONTR,  ID_EMPRESA, ID_CARTEIRA, PRINCIPAL, FX_CEP, atraso, IDADE, perfixo, TEL_CONF, PG, CC, STATUS_ENDERECO, STATUS_EMAIL, CPC, TRA, AUT

#   onde:
#       FX_CEP = 0 quando ha valor missing
#       IDADE = ROUND((DATA_DE_HOJE - DATA_NASCIMENTO)/365.25, 2)
#       STATUS_EMAIL = 0 quando ha valor missing

#
# ______________________________________________________________________

# ______________________________________________________________________
# IMPORTING LIBRARIES

import pandas
import numpy
import sys
import scipy
import matplotlib
import sklearn

from openpyxl import load_workbook
import xlsxwriter
import xlrd
import win32com.client as win32


# ______________________________________________________________________
# IMPORTING MODULES, FUNCIONS AND OBJECTS

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# theano
import theano
# print('theano: %s' % theano.__version__)
# tensorflow
import tensorflow
# print('tensorflow: %s' % tensorflow.__version__)
# keras
import keras
# print('keras: %s' % keras.__version__)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import max_norm
from keras.optimizers import SGD
from keras.models import model_from_json


# ______________________________________________________________________________________
# LOADING DATABASES

# datasets
# path_out = r"C:\Users\GKrat1\Documents\9. Python Projects\Fattor\4. DB 20181105\Developing\Emp30\Temp Output.csv"
path0 = r"C:\Users\GKrat1\Documents\9. Python Projects\Fattor\4. DB 20181105\Developing\DEV181001-181007_Emp30_14Var_ID_CONTR.csv"
# path0 = r"C:\Users\GKrat1\Documents\9. Python Projects\Fattor\4. DB 20181105\Developing\Emp30\DEV181001-181007_Emp30_14Var_ID_CONTR.xlsx"
# path1 = r"C:\Users\GKrat1\Documents\9. Python Projects\Fattor\4. DB 20181105\Developing\Emp30\SCO181008-181028_Emp30_15Var_CND NoGBI.csv"

seed = 7
validation_size = 0.20


# ______________________________________________________________________
# BUILDING NETWORK (MACRO)


def macro_model(nrounds, nneurons, nneurons2, nneurons3, nneurons4, X_train, Y_train):

    def run_models(neurons, neurons2, neurons3, neurons4, initializer, epochs, batch_size, prediction_name, weights_name):

        # create model
        model = Sequential()
        model.add(Dense(neurons, input_dim=neurons, kernel_initializer=initializer, activation='relu'))
        # model.add(Dense(neurons2, input_dim=neurons2, kernel_initializer=initializer, activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(neurons2, input_dim=neurons2, kernel_initializer=initializer, activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(neurons, input_dim=neurons, kernel_initializer=initializer, activation='relu'))
        # model.add(Dropout(0.2))
        # model.add(Dense(neurons3, input_dim=neurons3, kernel_initializer=initializer, activation='relu'))
        # model.add(Dropout(0.2))
        # model.add(Dense(neurons4, input_dim=neurons4, kernel_initializer=initializer, activation='relu'))
        # model.add(Dropout(0.2))
        model.add(Dense(1, kernel_initializer=initializer, activation='sigmoid'))

        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
        # model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # Fit the model
        model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)

        # evaluate the model
        scores = model.evaluate(X_train, Y_train)
        print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

        # serialize model to JSON
        model_json = model.to_json()
        with open(prediction_name, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(weights_name)
        print("Saved model to disk")

        # # calculate predictions
        # predictions = model.predict(X_validation)
        # # print(predictions)

        # model_results = pandas.DataFrame(predictions)
        # all_results.append(model_results)

        # prediction_df.append(model_results)

    prediction_name_list = ['RES_P1', 'RES_P2', 'RES_P3', 'RES_P4', 'RES_P5', 'RES_P6', 'RES_P7', 'RES_P8', 'RES_P9', 'RES_P10', 'RES_P11', 'RES_P12', 'RES_P13', 'RES_P14', 'RES_P15', 'RES_P16', 'RES_P17', 'RES_P18', 'RES_P19', 'RES_P20']
    weights_name_list = ['RES_P1.h5', 'RES_P2.h5', 'RES_P3.h5', 'RES_P4.h5', 'RES_P5.h5', 'RES_P6.h5', 'RES_P7.h5', 'RES_P8.h5', 'RES_P9.h5', 'RES_P10.h5', 'RES_P11.h5', 'RES_P12.h5', 'RES_P13.h5', 'RES_P14.h5', 'RES_P15.h5', 'RES_P16.h5', 'RES_P17.h5', 'RES_P18.h5', 'RES_P19.h5', 'RES_P20.h5']
    initializer_list = ['random_normal', 'random_normal', 'random_uniform', 'random_uniform', 'normal', 'normal', 'uniform', 'uniform', 'random_normal', 'random_uniform', 'normal', 'uniform', 'random_normal', 'random_normal', 'random_uniform', 'random_uniform', 'normal', 'normal', 'uniform', 'uniform']
    # epochs_list = [300, 400, 300, 400, 300, 400, 300, 400, 200, 200, 200, 200, 150, 50, 150, 50, 150, 50, 150, 50]
    # batch_size_list = [70, 90, 70, 90, 70, 90, 70, 90, 50, 50, 50, 50, 40, 20, 40, 20, 40, 20, 40, 20]
    epochs_list = [300, 400, 300, 400, 300, 400, 300, 400, 200, 200, 200, 200, 150, 50, 150, 50, 150, 50, 150, 50]
    batch_size_list = [70, 90, 70, 90, 70, 90, 70, 90, 50, 50, 50, 50, 40, 20, 40, 20, 40, 20, 40, 20]

    all_results = []
    c = nrounds
    j = 0
    while(j < c):
        prediction_name = prediction_name_list[j]
        weights_name = weights_name_list[j]
        initializer = initializer_list[j]
        epochs = epochs_list[j]
        batch_size = batch_size_list[j]

        neurons = nneurons
        neurons2 = nneurons2
        neurons3 = nneurons3
        neurons4 = nneurons4
        run_models(neurons, neurons2, neurons3, neurons4, initializer, epochs, batch_size, prediction_name, weights_name)

        print(prediction_name)

        j = j + 1


# ______________________________________________________________________________________
# TRAINING MODELS

dataset10 = pandas.read_csv(path0)
# dataset10 = pandas.read_excel(path0)
array10 = dataset10.values

X_train = array10[:, 2:16]
Y_train = array10[:, 16]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

response_score = macro_model(8, 14, 28, 7, 3, X_train, Y_train)
# response_score = response_score.join(db_ysco20)


# db9 = response_score

# ______________________________________________________________________
# RECORDING PREDICTIONS

# # db_name = 'FILE NAME'

# # wb = load_workbook(path_out, keep_vba=False)
# # writer = pandas.ExcelWriter(path_out, engine='openpyxl')
# # writer.book = wb
# # writer.sheets = dict((ws.title, ws) for ws in wb.worksheets)

# # db9.to_excel(writer, db_name, index=True, index_label='Index', header=True)
# # writer.save()


# db9.to_csv(path_or_buf=path_out)
