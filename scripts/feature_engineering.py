import os
import pandas as pd
from utils import DataLoader, DataSaver, Scaler
from custompreprocessor import CustomPreprocessor
from sklearn.preprocessing import MinMaxScaler


def engineer_features(input_file, output_file, scaler_path):
    loader = DataLoader(input_file)
    df = loader.load_data()

    # Convertir las columnas 'sourcePort' y 'destinationPort' a int64
    df['sourcePort'] = pd.to_numeric(df['sourcePort'], errors='coerce').fillna(0).astype('int64')
    df['destinationPort'] = pd.to_numeric(df['destinationPort'], errors='coerce').fillna(0).astype('int64')

    # Corregir valores negativos en las columnas de bytes y paquetes
    columns_to_check = ['totalSourceBytes', 'totalDestinationBytes', 'totalSourcePackets', 'totalDestinationPackets']
    for col in columns_to_check:
        df[col] = df[col].apply(lambda x: max(x, 0))  # Convertir valores negativos a 0

    # Eliminar columnas irrelevantes
    cols_to_drop = [
        'generated', 'appName', 'sourcePayloadAsBase64',
        'sourcePayloadAsUTF', 'destinationPayloadAsBase64', 
        'destinationPayloadAsUTF'
    ]
    df = df.drop(columns=cols_to_drop)

    # Aplicar preprocesador personalizado
    preprocessor = CustomPreprocessor()
    df = preprocessor.fit_transform(df)

    # Mantener la columna 'Label' separada para que no se procese en OHE
    label = df['Label']
    df = df.drop(columns=['Label'])

    # Aplicar One-Hot Encoding a las columnas categóricas restantes
    df = pd.get_dummies(df, drop_first=True)

    # Reintegrar la columna 'Label'
    df['Label'] = label

    # Escalar características numéricas
    scaler = Scaler()
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.difference(columns_to_check)
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Usar MinMaxScaler para las columnas que no deben tener valores negativos
    min_max_scaler = MinMaxScaler()
    df[columns_to_check] = min_max_scaler.fit_transform(df[columns_to_check])

    # Guardar el scaler de las características restantes
    scaler.save(scaler_path)

    # Guardar los datos procesados
    saver = DataSaver(output_file)
    saver.save_data(df)

    print(df.shape)  # Cuántos registros hay 
    print(df.info())

if __name__ == "__main__":
    engineer_features("data/processed/prepared_data.csv", "data/processed/engineered_data.csv", "data/models/scaler.pkl")
