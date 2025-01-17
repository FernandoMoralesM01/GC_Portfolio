import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import date
import yfinance as yf
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import load_model
from statsmodels.tsa.stattools import grangercausalitytests
import tools.modeltools as mt
import warnings

warnings.filterwarnings("ignore")


def realizagranger(df, maxlag=6, epsilon=0.01, sel1="HRV", sel2="HRV"):
    try:
        if len(df) <= maxlag:
            print(f"{len(df)} Datos insuficientes!")
            return 0
        
        # Realizar la prueba de causalidad de Granger
        gc_test_1 = grangercausalitytests(df[[sel1, sel2]], maxlag=maxlag, verbose=False);
        p_values = [gc_test_1[i + 1][0]['ssr_chi2test'][1] for i in range(maxlag)]
        
        # Verificar si la media de los p-valores es menor que el umbral
        for val in p_values:
            if val < epsilon:
                return 1
        return 0
    except Exception as e:
        #print(f"Error: {e}")
        return 0
    

def get_vector_causalidad(df):
    vector = []
    for col in df.columns.values:
        if col == df.columns[0]:
            continue
        else:
            vector.append(realizagranger(df, sel1=df.columns[0], sel2=col, epsilon=0.05, maxlag=3))
    return np.array(vector)

def get_features(activo, path_to_activos):

    df_precios = pd.DataFrame()
    activos = pd.read_csv(path_to_activos)['Activos'].tolist()
    
    activos = [i for i in activos if i != activo]
    activos = [activo] + activos
    
    n = len(activos)
    today = date.today()
    start = today - timedelta(days=40)

    for i in activos:
        df_precios[i] = pd.DataFrame(yf.Ticker(i).history(start = start,end = today )['Close'])
    
    df_precios.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_precios.interpolate(method='linear', inplace=True)
    
    print("Creando la matriz de causalidad")
    vector = get_vector_causalidad(df_precios)
    activos_influyentes_index = np.where(vector > 0)[0] + 1
    activos_influyentes = df_precios.columns[activos_influyentes_index].values.tolist()
    return activos_influyentes


def get_time_series(node, connected_nodes):
    df_rasgos = pd.DataFrame()

    activos_seleccionados = [node] + connected_nodes

    today = date.today()
    start = today - timedelta(days=300)

    print(f"Obteniendo las series de tiempo de {activos_seleccionados}")
    for i in activos_seleccionados:
        df_rasgos[i] = pd.DataFrame(yf.Ticker(i).history(start = start,end = today, )['Close'])   
        df_rasgos[i] = (df_rasgos[i] - df_rasgos[i].mean()) / df_rasgos[i].std()
    #df_rasgos.interpolate(method='linear', inplace=True)
    df_rasgos.fillna(method='ffill', inplace=True)
    df_rasgos.fillna(method='bfill', inplace=True)
    return df_rasgos

def get_model_from_gc(activo, path_to_activos, path_to_models, model = "Bi-LSTM"):
    nombre_modelo = f"{model}_{activo}.keras"
    model_path = fr"{path_to_models}\{nombre_modelo}"
    ## Obtener los rasgos del activo

    activos_influyentes = get_features(activo, path_to_activos)[:5]
    df_features = get_time_series(activo, activos_influyentes)
    
    X_train, y_train, X_test, y_test = mt.genera_grupos_activos(df_features, len_sec=20, train_split=0.2, output=activo, step=4, random=True, future_win = 4)

    model_ = mt.get_model(X_train, y_train)
    
    try:
        model = load_model(model_path)
    except:
        print("Error: No se encontro el modelo, se creara uno nuevo")
    
    model, history = mt.train_model(path = model_path, model=model_, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, epochs=10, batch_size=32, lr=0.001)

    return model, history, activos_influyentes
