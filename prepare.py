import lightgbm as lgb
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, precision_score, recall_score
import joblib
import pandas as pd
import numpy as np
import threading
import time
import signal
import sys

from prometheus_client import start_http_server, Gauge
import time

# Определение метрик
f1_score_gauge = Gauge('f1_score', 'F1 Score of the model')
precision_score_gauge = Gauge('precision_score', 'Precision Score of the model')
recall_score_gauge = Gauge('recall_score', 'Recall Score of the model')

# Обработчик сигнала для корректного завершения работы
def signal_handler(sig, frame):
    print('Получен сигнал завершения, завершаем сервер...')
    sys.exit(0)

# Оценка модели
def scors(gbm,y_test, y_pred):
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted',zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted')
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    score_train = dict([(score[1], score[2]) for score in gbm.eval_train()])
    print('The score of the current model in the training set is: multi_logloss=%.4f, multi_error=%.4f, \n'
        % (score_train['multi_logloss'], score_train['multi_error']))

    f1_score_gauge.set(f1)
    precision_score_gauge.set(precision)
    recall_score_gauge.set(recall)



def streaming_reading(X_train, y_train, batch_size=500):
    X = []
    y = []
    current_line = 0
    train_data, train_label = shuffle(X_train, y_train, random_state=0)
    train_data = train_data.to_numpy()
    for row, target in zip(train_data, train_label):
        X.append(row)
        y.append(target)

        current_line += 1
        if current_line >= batch_size:
            X, y = np.array(X), np.array(y)
            yield X, y
            X, y = [], []
            current_line = 0
    X, y = np.array(X), np.array(y)
    yield X, y

def IncrementaLightGbm(X, y,numclass):  
    gbm = None

    params = {
        'objective': 'multiclass',
        'num_class': numclass, # Обновлено в соответствии с количеством уникальных классов
        'metric': ['multi_logloss', 'multi_error'],
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'num_leaves': 31,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.5,
        'bagging_freq': 20,
        'learning_rate': 0.05,
        'verbose': 0
    }    
    streaming_train_iterators = streaming_reading(X, y, batch_size=500)

    for i, data in enumerate(streaming_train_iterators):
        print(f"iteration: {i}")
        X_batch = data[0]
        y_batch = data[1]
        X_train, X_test, y_train, y_test = train_test_split(X_batch, y_batch, test_size=0.1, random_state=0)
        y_train = y_train.ravel()
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=1000,
                        valid_sets=lgb_eval,
                        init_model=gbm, 
                        # early_stopping_rounds=10,
                        # verbose_eval=False,
                        keep_training_booster=True)  

        # Предсказание на тестовой выборке
        y_pred = gbm.predict(X_test)
        # lgb_f1_score(y_test,y_pred)
        y_pred = np.argmax(y_pred, axis=1)
        # # Преобразование предсказанных метк обратно к оригинальным значениям
        # y_pred_original = [label_mapping[pred] for pred in y_pred]

        scors(gbm,y_test, y_pred)

    return gbm

# Функция для запуска HTTP сервера
def start_metrics_server(port):
    start_http_server(port)
    while True:
        time.sleep(1)

def update_metrics():
        # load original first data
    df = pd.read_csv('winequality-red.csv',delimiter=";")
    #возможные градации качества вина
    df.value_counts("quality")
    X = df.drop('quality', axis = 1)
    y = df['quality']
    numclass=df["quality"].nunique() #count of unique classes for model eval
    # Преобразование метки класса так, чтобы они начинались с 0
    y_encoded, y_labels = pd.factorize(y)
    # Создание словаря для соответствия между факторизованными метками и оригинальными
    label_mapping = dict(zip(range(len(y_labels)), y_labels))
    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    while True:
        # Здесь вычисляются новые значения метрик
        gbm = IncrementaLightGbm(X_train, y_train,numclass)
        y_pred = gbm.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        scors(gbm,y_test, y_pred)
        time.sleep(1) # Обновляем метрики каждые 10 секунд

if __name__ == '__main__':
    # Установка обработчика сигнала
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Запуск сервера метрик на порту 8000 в отдельном потоке
    server_thread = threading.Thread(target=start_metrics_server, args=(8000,))
    server_thread.start()

    # Запуск обновления метрик в отдельном потоке
    metrics_thread = threading.Thread(target=update_metrics)
    metrics_thread.start()

    # Ожидание завершения обновления метрик
    metrics_thread.join()

