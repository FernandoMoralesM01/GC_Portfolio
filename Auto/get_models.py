import pandas as pd
import tools.pipeline_getmodel as gm
import os

def main():
    df_portafolio = pd.read_csv(r"C:\Users\fercy\OneDrive\Documentos\GC_Portfolio\Notebooks\otros\portaolio_activos_1.csv")
    df_portafolio_influyentes = df_portafolio.copy()
    path_all_activos = r"C:\Users\fercy\OneDrive\Documentos\GC_Portfolio\Notebooks\otros\activos.csv"
    path_models = r"C:\Users\fercy\OneDrive\Documentos\GC_Portfolio\Auto\models"
    
    for activo in df_portafolio["Activos"]:
        model, history, influentes = gm.get_model_from_gc(activo, path_all_activos, path_models, model="Bi-LSTM")
        df_portafolio_influyentes.loc[df_portafolio_influyentes["Activos"] == activo, "Activos_Influyentes"] = ','.join(influentes)
        print(f"Modelo de {activo} descargado")
    
    df_portafolio_influyentes.to_csv(r"C:\Users\fercy\OneDrive\Documentos\GC_Portfolio\Notebooks\otros\portaolio_activos_1_influyentes.csv", index=False)
    
    print("Downloading models...")

if __name__ == "__main__":
    main()