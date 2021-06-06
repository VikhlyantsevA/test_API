from flask import Flask, request, jsonify, make_response
import traceback
import pickle
import random
import pandas as pd
import numpy as np
import re
import os

from train import run_training

app = Flask(__name__)
   
@app.route('/get_prediction', methods=['GET', 'POST'])
def get_prediction():
    ##Обучение моделей. Если модели уже обучены и сохранены, run_training ничего не выполняет.
    run_training()

    ## Чтение всех файлов и папок, в том числе вложенных, поиск моделей.
    for root, directory, files_names in os.walk('models'):
        models_names = [file_name.replace('.pkl','') for file_name in files_names if re.match(r'^model_\d{2}\.pkl$', file_name)]
        
    ## Импорт моделей.
    for model_name in models_names:
        with open(f'models/{model_name}.pkl', 'rb') as f:
           locals()[model_name] = pickle.load(f)
    
    ## Импорт названий колонок.   
    with open('models/columns_names.pkl', 'rb') as f:
       columns_names = pickle.load(f)
    
    ##Обработка ошибок.
    for key, value in request.args.items():
        if key not in columns_names:
            return f'''<h1> Передан несуществующий параметр. </h1>
                       <h2> key = {key} </h2>'''
        try:
            float(value)
        except:
            return f'''<h1> Недопустимый тип данных.</h1>
                       <h2> key = {key}, value = {value} </h2>
                       <p>{traceback.format_exc()}</p>'''
    
    ## Обработка запроса, заполнение недостающих параметров.
    columns_values = [[float(request.args.get(column_name, random.uniform(-10**6, 10**6)))] for column_name in columns_names]
    X_to_pred_dict = dict(zip(columns_names, columns_values))
    ## Подстановка параметров в модели для прогнозирования страты
    Y_pred_dict = {}
    for model_name in models_names:
        Y_pred_dict[model_name] = list(map(int, eval(model_name).predict(pd.DataFrame(X_to_pred_dict))))
    X_to_pred_rounded = {key:list(np.around(np.array(value),4)) for key, value in X_to_pred_dict.items()}
    return jsonify([X_to_pred_rounded, Y_pred_dict])

    ##То же что и выше, но немного другим способом    
    # columns_values = np.array([[request.args.get(column_name, random.uniform(-10**6, 10**6)) for column_name in columns_names]], dtype = float)
    # X_to_pred = pd.DataFrame(data = columns_values, columns = columns_names)
    # Y_pred_dict = {}
    # for model_name in models_names:
        # Y_pred_dict[model_name] = eval(model_name).predict(X_to_pred)
    # Y_pred = pd.DataFrame(Y_pred_dict)
    # return jsonify([X_to_pred.round(4).to_dict(), Y_pred.to_dict()])

if __name__ == '__main__':
    ## Запуск приложения
    app.run(debug=True, port=5000)

# Пример запроса
# http://127.0.0.1:5000/get_prediction?p02=123&p03=1&p04=5&p05=5&p06=34&p07=23&p08=45&p09=12&p10=345