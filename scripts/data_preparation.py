import os
import pandas as pd
from utils import DataLoader, DataSaver

def assign_labels_by_day(df):
    # Mapear el día a la etiqueta correspondiente
    day_to_label = {
        'mon': 'HTTPDoS',
        'tue': 'DDos',
        'thu': 'SSHBruteForce',
        'sun': 'Infiltration',
        'sat': 'Normal',  
        'wed': 'Normal'   
    }
    
    # Asignar etiquetas: si es 'Attack', convertir a la etiqueta correspondiente; si no, mantener la etiqueta actual
    df['Label'] = df.apply(lambda row: day_to_label.get(row['day'], 'Normal') if row['Label'] == 'Attack' else row['Label'], axis=1)

    # Por inverosimilitud en la documentación elimino estos registros en los días de tráfico 'normal'
    df = df[df['Label'] != 'Attack']

    return df

def extract_day_from_filename(filename):
    # Extraer el día basado en el nombre del archivo
    if "Mon" in filename:
        return "mon"
    elif "Tue" in filename:
        return "tue"
    elif "Thu" in filename:
        return "thu"
    elif "Sun" in filename:
        return "sun"
    else:
        return "other"  # Por si hay otros días no contemplados

def clean_data(df):
    # Eliminar logs con la IP 0.0.0.0
    df = df[(df['source'] != '0.0.0.0') & (df['destination'] != '0.0.0.0')]

    # Eliminar logs con la dirección R2R
    df = df[(df['direction'] != 'R2R')]

    # Limpiar las columnas TCPFlagsDescription
    valid_flags = {'A', 'F', 'S', 'R', 'P'}
    df['sourceTCPFlagsDescription'] = df['sourceTCPFlagsDescription'].apply(
        lambda x: ''.join([char for char in str(x) if char in valid_flags]) if pd.notna(x) else x
    )
    df['destinationTCPFlagsDescription'] = df['destinationTCPFlagsDescription'].apply(
        lambda x: ''.join([char for char in str(x) if char in valid_flags]) if pd.notna(x) else x
    )
    # Eliminar la columna day 
    df = df.drop('day', axis=1)

    return df

def merge_csv_files(input_dir, output_file):
    files = [
        "TestbedSatJun12Flows.csv",   # Normal
        "TestbedSunJun13Flows.csv",   # Infiltration
        "TestbedMonJun14Flows.csv",   # HTTPDoS
        "TestbedTueJun15Flows.csv",   # DDos
        "TestbedWedJun16Flows.csv",   # Normal (o lo que corresponda)
        "TestbedThuJun17Flows.csv"    # SSHBruteForce
    ]
    df_list = []
    for file in files:
        df = pd.read_csv(os.path.join(input_dir, file))
        day = extract_day_from_filename(file)  # Extraer el día usando la nueva función
        df['day'] = day  # Añadir una columna para identificar el día
        df_list.append(df)
    df_merged = pd.concat(df_list, ignore_index=True)
    df_merged.to_csv(output_file, index=False)
    return df_merged

def prepare_data(input_dir, output_file):
    df = merge_csv_files(input_dir, output_file)
    
    # Asignar etiquetas basadas en el día de la semana
    df = assign_labels_by_day(df)
    
    # Limpiar datos según las reglas especificadas
    df = clean_data(df)

    saver = DataSaver(output_file)
    saver.save_data(df)

if __name__ == "__main__":
    prepare_data("data/raw/", "data/processed/prepared_data.csv")

# Verificación de los resultados
df = pd.read_csv("data/processed/prepared_data.csv")
print(df['Label'].value_counts())  # Cuántos registros hay para cada etiqueta
print(df.info())
