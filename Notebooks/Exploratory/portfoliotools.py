import numpy as np
import pandas as pd
import math

def calcular_estadisticas(df_precios):
    """
    Calcula estadísticas relevantes para un DataFrame de precios.

    Args:
        df_precios (pd.DataFrame): DataFrame con precios históricos de activos.

    Returns:
        dict: Diccionario con rendimiento medio, varianza, desviación estándar, 
              matriz de covarianza y matriz de correlación anualizadas.
    """
    df_precios_ren = np.log(df_precios).diff().dropna()
    estadisticas = {
        'Media': df_precios_ren.mean().values * 252,
        'Varianza': df_precios_ren.var().values * 252,
        'DesvEst': np.sqrt(df_precios_ren.var().values * 252),
        'matriz_cov': df_precios_ren.cov() * 252,
        'matriz_corr': df_precios_ren.corr() * 252
    }
    return estadisticas

def crear_portafolio(activos, ultimo_precio, pesos, capital):
    """
    Crea un DataFrame con la composición del portafolio.

    Args:
        activos (array): Nombres de los activos.
        ultimo_precio (array): Precios actuales de los activos.
        pesos (array): Pesos de los activos en el portafolio.
        capital (float): Capital total disponible.

    Returns:
        pd.DataFrame: DataFrame con la composición del portafolio.
    """
    montos = [capital * w for w in pesos]
    n_activos = [math.floor(m / p) for m, p in zip(montos, ultimo_precio)]
    monto_real = [n * p for n, p in zip(n_activos, ultimo_precio)]

    portafolio = pd.DataFrame(
        zip(activos, ultimo_precio, n_activos, monto_real, pesos),
        columns=['Activos', 'Precio', '# Activos', 'Monto', '% Capital']
    )
    portafolio['Precio'] = portafolio['Precio'].apply(lambda x: x)
    portafolio['Monto'] = portafolio['Monto'].apply(lambda x: x)
    portafolio['% Capital'] = portafolio['% Capital'].apply(lambda x: x * 100)

    return portafolio

def calcular_portafolio(df_precios, capital, objetivo='min_var', verbose=True, umbral=0.05):
    """
    Calcula el portafolio basado en el criterio de mínima varianza o máximo Sharpe.

    Args:
        df_precios (pd.DataFrame): DataFrame con precios históricos de activos.
        capital (float): Capital total disponible.
        objetivo (str): Criterio de optimización ('min_var' o 'max_sharpe').
        verbose (bool): Si True, imprime los resultados.

    Returns:
        pd.DataFrame: Composición del portafolio.
    """
    activos = df_precios.columns.values
    n = len(activos)
    estadisticas = calcular_estadisticas(df_precios)
    matriz_cov = estadisticas['matriz_cov']
    Media = estadisticas['Media']

    vector_unos = np.ones(n)

    if objetivo == 'min_var':
        pesos = (np.linalg.inv(matriz_cov) @ vector_unos) / (vector_unos @ np.linalg.inv(matriz_cov) @ vector_unos)
        descripcion = "Portafolio Mínima Varianza"
    elif objetivo == 'max_sharpe':
        pesos = (np.linalg.inv(matriz_cov) @ Media) / (vector_unos @ np.linalg.inv(matriz_cov) @ Media)
        descripcion = "Portafolio Máximo Sharpe"
    else:
        raise ValueError("El objetivo debe ser 'min_var' o 'max_sharpe'")

    ultimo_precio = df_precios.iloc[-1].values
    portafolio = crear_portafolio(activos, ultimo_precio, pesos, capital)

    if verbose:
        print(descripcion)
        print(portafolio)
        print('\nEl monto real a invertir es de', f"${np.sum(portafolio['Monto'].str.replace('$','').astype(float)):.2f}")

        rendimiento = Media @ pesos
        volatilidad = np.sqrt(pesos @ matriz_cov @ pesos)
        sharpe = rendimiento / volatilidad

        print(f"Rendimiento esperado: {rendimiento * 100:.2f}%")
        print(f"Volatilidad: {volatilidad:.4f}")
        print(f"Cociente de Sharpe: {sharpe:.4f}")

    return portafolio



def calcular_rendimiento_volatilidad(Media, matriz_cov, w):
    ren_port = Media @ w
    vol_port = np.sqrt(w @ matriz_cov @ w)
    sharpe_port = ren_port / vol_port
    return ren_port, vol_port, sharpe_port

def calcular_rendimiento_volatilidad_extendido(Media, matriz_cov, w_port_min_var, w_port_max_sharpe):
    vector_unos = [1] * len(Media)

    A = vector_unos @ np.linalg.inv(matriz_cov) @ vector_unos
    B = vector_unos @ np.linalg.inv(matriz_cov) @ Media
    C = Media @ np.linalg.inv(matriz_cov) @ Media

    ren_deseado = np.arange(0, 1.01, 0.01)
    mat = np.zeros((len(w_port_min_var), len(ren_deseado)))

    v_LA = []
    v_nB = []

    for i in range(len(ren_deseado)):
        L = (C - B * ren_deseado[i]) / (A * C - B**2)
        LA = L * A
        nB = 1 - LA
        v_LA.append(LA)
        v_nB.append(nB)

    for k in range(len(w_port_min_var)):
        for i in range(len(ren_deseado)):
            mat[k, i] = v_LA[i] * w_port_min_var[k] + v_nB[i] * w_port_max_sharpe[k]

    mat = mat.T
    ren_port_min_esp = [i @ Media for i in mat]
    vol_port_min_esp = np.sqrt([i @ matriz_cov @ i for i in mat])

    return np.stack([ren_port_min_esp, vol_port_min_esp], axis=1)
