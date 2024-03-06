# создание сервера для обновления метрик при дообучении модели
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

# библиотеки для связи с прометеус
from prometheus_client import start_http_server, Gauge
import time

# Определение метрик, которые будут апдейтиться в прометеус
f1_score_gauge = Gauge('f1_score', 'F1 Score of the model')
precision_score_gauge = Gauge('precision_score', 'Precision Score of the model')
recall_score_gauge = Gauge('recall_score', 'Recall Score of the model')

# Обработчик сигнала для корректного завершения работы
def signal_handler(sig, frame):
    print('Получен сигнал завершения, завершаем сервер...')
    sys.exit(0)

# Оценка модели
def scors(gbm,y_test, y_pred):
    # вывод метрик при обучении модели
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted',zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted')
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    score_train = dict([(score[1], score[2]) for score in gbm.eval_train()])
    print('The score of the current model in the training set is: multi_logloss=%.4f, multi_error=%.4f, \n'
        % (score_train['multi_logloss'], score_train['multi_error']))

    # , а так же передача этих метрик на сервер прометеус
    f1_score_gauge.set(f1)
    precision_score_gauge.set(precision)
    recall_score_gauge.set(recall)



def streaming_reading(X_train, y_train, batch_size=500):
    # спецзаглушка, чтобы эмулировать постепенный приход новых данных для дообучения
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
    # модель для мультиклассификации 
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

    # создание итератора, эмулирующего постепенный приход очередного чанка новых данных для дообучения
    streaming_train_iterators = streaming_reading(X, y, batch_size=500)

    for i, data in enumerate(streaming_train_iterators):
        print(f"iteration: {i}")
        X_batch = data[0]
        y_batch = data[1]
        X_train, X_test, y_train, y_test = train_test_split(X_batch, y_batch, test_size=0.1, random_state=0)
        y_train = y_train.ravel()
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
        # дообучение модели на очередной порции данных
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

        # обновление метрик в т.ч. для передачи в прометеус
        scors(gbm,y_test, y_pred)

    return gbm

# Функция для запуска HTTP сервера, который принимает обновленные метрики и будет их предоставлять прометеусу
def start_metrics_server(port):
    start_http_server(port)
    while True:
        time.sleep(1)

def update_metrics():
    # подготовительные действия для загрузки исходных данных по винам
        # load original first data
    df = pd.read_csv('winequality-red.csv',delimiter=";")
    #возможные градации качества вина
    df.value_counts("quality")
    X = df.drop('quality', axis = 1)
    y = df['quality']
    numclass=df["quality"].nunique() #количество классов
    # Преобразование метки класса так, чтобы они начинались с 0 - это потребовалось для работы модели
    # изначально метки были не с 0, и это не работало
    y_encoded, y_labels = pd.factorize(y)
    # Создание словаря для соответствия между факторизованными метками и оригинальными
    label_mapping = dict(zip(range(len(y_labels)), y_labels)) #это не используется - для обратного раскодирования меток
    
    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    while True:
        # Здесь вычисляются новые значения метрик
        # для эмуляции мониторинга дообучения модели и отображения метрик в прометеус/графане выполняется бесконечный цикл
        # на имеющихся данных модель дообучается на кусках, пока все данные не исчерпаются, потом
        # процесс запускается сначала.
        # в реальности предполагается поступление все новых и новых данных
        # в данном случае модель обрабатывает весь датасет за три раза и выходит из IncrementaLightGbm, чтобы 
        # потом начать сначала
        gbm = IncrementaLightGbm(X_train, y_train,numclass)
        y_pred = gbm.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        scors(gbm,y_test, y_pred)
        time.sleep(1) 

if __name__ == '__main__':
    # Установка обработчика сигнала для завершения потоков
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # запускаются два параллельные потока,
    # один поток - сервер прокладка, который предоставляет метрики для прометеуса на порту 8000
    # второй поток - модель, которая дообучается и сообщает в процессе текущие метрики

    # Запуск сервера метрик на порту 8000 в отдельном потоке
    server_thread = threading.Thread(target=start_metrics_server, args=(8000,))
    server_thread.start()

    # Запуск обновления метрик в отдельном потоке
    metrics_thread = threading.Thread(target=update_metrics)
    metrics_thread.start()

    # Ожидание завершения обновления метрик
    metrics_thread.join()

