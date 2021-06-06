from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import pandas as pd
import pickle
import os

def training_performed(clusters_numbers):
    ## Проверка прошли ли обучение модели
    if not os.path.exists('models/'):
        os.mkdir('models')
        return False
    else:
        for i in range(1, len(clusters_numbers) + 1):
            if not os.path.isfile(f'models/model_{i}.pkl'): 
                return False
        if not os.path.isfile(f'models/columns_names.pkl'):
                return False
        return True
    
def run_training():
    clusters_numbers = [10, 15, 20]
    if not training_performed(clusters_numbers):
        ## Создание тренировочного датасета: 
        ## X_train_arr - массив значений параметров, 
        ## Y_train_arr - значения целевого показателя (номер страты объекта)
        X_train_arr, Y_train_arr = make_blobs(n_samples=2000, centers=20, n_features=10)
        
        ## {i:02d} - при таком формате записи модели и параметров их вывод и расположение в директории осуществляется в порядке нумерации 
        parameters_number = X_train_arr.shape[1]
        columns_names = [f'p{i:02d}' for i in range(1, parameters_number + 1)]

        X_train = pd.DataFrame(data=X_train_arr, columns=columns_names)
        Y_train = pd.Series(Y_train_arr)

        ## Обучение и сохранение моделей с количеством кластеров из списка clusters_numbers
        with open(f'models/columns_names.pkl','wb') as f:
            pickle.dump(columns_names, f)

        for i, clusters_number in enumerate(clusters_numbers):
            model_i = KMeans(n_clusters=clusters_number, random_state=0).fit(X_train)
            with open(f'models/model_{(i + 1):02d}.pkl','wb') as f:
                pickle.dump(model_i, f)
