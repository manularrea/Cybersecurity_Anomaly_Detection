import pandas as pd 
import numpy as np
import struct
import socket
import re
import ipaddress




def writer_attack(df, day):
    """
    Actualiza la columna 'Label' de un DataFrame según el día de la semana.

    Parámetros:
    df : pd.DataFrame
        DataFrame con una columna 'Label'.
    day : str
        Día de la semana ('monday', 'tuesday', 'thursday', 'sunday').

    Modifica:
    La columna 'Label' en el DataFrame, cambiando 'Attack' a un valor específico:
    - 'monday' -> 'HTTPDoS'
    - 'tuesday' -> 'DDos'
    - 'thursday' -> 'SSHBruteForce'
    - 'sunday' -> 'Infiltration'
    """
        
    if day == 'monday':
        df.loc[df['Label'] == 'Attack', 'Label'] = 'HTTPDoS'
    elif day == 'tuesday':
        df.loc[df['Label'] == 'Attack', 'Label'] = 'DDos'
    elif day == 'thursday':
        df.loc[df['Label'] == 'Attack', 'Label'] = 'SSHBruteForce'
    elif day == 'sunday':
        df.loc[df['Label'] == 'Attack', 'Label'] = 'Infiltration'
    return df



def delete_wrong_logs(df):
    # Eliminar registros donde la columna 'Label' es igual a 'Attack' 
    df = df[df['Label'] != 'Attack']
    return df



def combine_datasets(dfs):
    """
    Combina una lista de DataFrames en uno solo.

    Parámetros:
    dfs (list of pd.DataFrame): Lista de DataFrames a combinar.

    Retorna:
    pd.DataFrame: DataFrame combinado.
    """
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df



def print_unique_values(df):
    """
    Imprime los valores únicos de cada columna en el DataFrame.

    Parámetros:
    df : pd.DataFrame
        DataFrame con los datos cargados.
    """
    for column in df.columns:
        unique_values = df[column].unique()
        print(f"Columna: {column}")
        print(f"Valores únicos ({len(unique_values)}): {unique_values}")
        print("-" * 50)



def drop_unnecessary_columns(df):
    """
    Elimina columnas que no son necesarias para el análisis.
    
    Parámetros:
    df : pd.DataFrame
        DataFrame con los datos cargados.
        
    Retorna:
    pd.DataFrame : DataFrame con las columnas innecesarias eliminadas.
    """
    df.drop(columns=['generated','sourcePayloadAsBase64', 'sourcePayloadAsUTF', 'destinationPayloadAsBase64', 'destinationPayloadAsUTF', 'appName'], inplace=True)
    return df

def drop_invalid_ip(df):
    """
    Elimina las filas de un DataFrame donde las direcciones IP en las columnas 
    'source' y 'destination' son '0.0.0.0'.

    Args:
        df (pandas.DataFrame): El DataFrame que contiene las columnas 'source' y 
        'destination'.

    Returns:
        pandas.DataFrame: El DataFrame filtrado sin las filas donde las direcciones IP 
        en las columnas 'source' o 'destination' son '0.0.0.0'.

    """ 
    df = df[(df['source']!= '0.0.0.0') | (df['destination']!= '0.0.0.0')]
    return df


def convert_ip_to_numeric(df):
    """
    Convierte direcciones IP en su valor numérico estándar
    
    Parámetros:
    df : pd.DataFrame
        DataFrame con los datos cargados.
        
    Retorna:
    pd.DataFrame : DataFrame con direcciones IP convertidas a valores numéricos.
    """
    df['source'] = df['source'].apply(lambda ip: struct.unpack("!I", socket.inet_aton(ip))[0] if isinstance(ip, str) else 0)
    df['destination'] = df['destination'].apply(lambda ip: struct.unpack("!I", socket.inet_aton(ip))[0] if isinstance(ip, str) else 0)
    return df


# Lista de siglas permitidas en las TCP Flags (solo la primera letra de cada flag)
tcp_flags = ['F', 'S', 'R', 'P', 'A']

# Función para filtrar y dejar solo las siglas permitidas en las columnas del DataFrame
def filter_tcp_flags_in_df(df):
    def filter_tcp_flags(description):
        if pd.isna(description):
            return np.nan
        # Filtrar las palabras que son siglas permitidas
        filtered_flags = [word for word in re.split(r'\W+', description) if word in tcp_flags]
        # Unir las palabras filtradas en una cadena o devolver NaN si no hay palabras válidas
        return ','.join(filtered_flags) if filtered_flags else np.nan

    # Aplicar la función a las columnas 'sourceTCPFlagsDescription' y 'destinationTCPFlagsDescription'
    df['sourceTCPFlagsDescription'] = df['sourceTCPFlagsDescription'].apply(filter_tcp_flags)
    df['destinationTCPFlagsDescription'] = df['destinationTCPFlagsDescription'].apply(filter_tcp_flags)

    return df

def drop_invalid_direction(df):
    df = df[(df['direction']!= 'R2R')]
    return df

def convert_dates(df):
    """
    Convierte columnas de fechas a tipo datetime y extrae características temporales.
    
    Parámetros:
    df : pd.DataFrame
        DataFrame con los datos cargados.
        
    Retorna:
    pd.DataFrame : DataFrame con columnas de fechas convertidas y características temporales añadidas.
    """
    # Convertir las columnas de fechas a tipo datetime
    df['startDateTime'] = pd.to_datetime(df['startDateTime'])
    df['stopDateTime'] = pd.to_datetime(df['stopDateTime'])
    
    # Extraer características temporales
    df['startMin'] = df['startDateTime'].dt.minute
    df['stopMin'] = df['stopDateTime'].dt.minute
    df['startHour'] = df['startDateTime'].dt.hour
    df['startDayOfWeek'] = df['startDateTime'].dt.dayofweek
    df['startMonth'] = df['startDateTime'].dt.month
    df['stopHour'] = df['stopDateTime'].dt.hour
    df['stopDayOfWeek'] = df['stopDateTime'].dt.dayofweek
    df['stopMonth'] = df['stopDateTime'].dt.month
    
    return df


def convert_datetime(df):
    """
    Convierte fechas a valores numéricos (timestamps Unix) para facilitar el análisis temporal.
    
    Parámetros:
    df : pd.DataFrame
        DataFrame con los datos cargados.
        
    Retorna:
    pd.DataFrame : DataFrame con fechas convertidas a timestamps Unix.
    """
    df['startDateTime'] = pd.to_datetime(df['startDateTime'], format="%m/%d/%Y %H:%M")
    df['stopDateTime'] = pd.to_datetime(df['stopDateTime'], format="%m/%d/%Y %H:%M")
    return df

def calculate_temporal_distance(df):	
    df['start_fractional_hour'] = df['startDateTime'].dt.hour + df['startDateTime'].dt.minute / 60
    df['stop_fractional_hour'] = df['stopDateTime'].dt.hour + df['stopDateTime'].dt.minute / 60

    df['start_hour_sin'] = np.sin(2 * np.pi * df['start_fractional_hour'] / 24)
    df['stop_hour_sin'] = np.sin(2 * np.pi * df['stop_fractional_hour'] / 24)

    df['start_hour_cos'] = np.cos(2 * np.pi * df['start_fractional_hour'] / 24)
    df['stop_hour_cos'] = np.cos(2 * np.pi * df['stop_fractional_hour'] / 24)

    return df

def convert_datetime_to_unix(df):
    """
    Convierte fechas a valores numéricos (timestamps Unix) para facilitar el análisis temporal.
    
    Parámetros:
    df : pd.DataFrame
        DataFrame con los datos cargados.
        
    Retorna:
    pd.DataFrame : DataFrame con fechas convertidas a timestamps Unix.
    """
    df['startDateTime'] = pd.to_datetime(df['startDateTime'], errors='coerce').astype('int64') // 10**9
    df['stopDateTime'] = pd.to_datetime(df['stopDateTime'], errors='coerce').astype('int64') // 10**9
    return df


def flags_transform(flags):
        
    """
    Transforma una cadena de caracteres que representa banderas de red (network flags) en un valor numérico 
    sumando los valores ASCII de sus caracteres.

    Parámetros:
    -----------
    flags : str
        Una cadena de caracteres que representa las banderas de red. Esta cadena puede contener letras y 
        comas, y los espacios serán eliminados antes de la transformación.
    
    Retorna:
    --------
    int
        El valor numérico resultante de la suma de los valores ASCII de los caracteres en la cadena de 
        entrada, excluyendo las comas y espacios.

    """
    value = 0
    if type(flags) is str:
        flags = flags.replace(" ", "")
        for c in flags:
            if c != ',':
                value += ord(c)
    return value



def transform_tcp_flags(df):
    """
    Transforma descripciones de flags TCP a valores numéricos.
    
    Parámetros:
    df : pd.DataFrame
        DataFrame con los datos cargados.
        
    Retorna:
    pd.DataFrame : DataFrame con descripciones de flags TCP transformadas a valores numéricos.
    """
    df['sourceTCPFlagsDescription'] = df['sourceTCPFlagsDescription'].apply(flags_transform)
    df['destinationTCPFlagsDescription'] = df['destinationTCPFlagsDescription'].apply(flags_transform)
    return df



def convert_categorical_columns(df):
    """
    Convierte columnas categóricas a códigos numéricos.
    
    Parámetros:
    df : pd.DataFrame
        DataFrame con los datos cargados.
        
    Retorna:
    pd.DataFrame : DataFrame con columnas categóricas convertidas a códigos numéricos.
    """
    df['protocolName'] = df['protocolName'].astype('category').cat.codes
    df['appName'] = df['appName'].astype('category').cat.codes
    df['direction'] = df['direction'].astype('category').cat.codes
    return df



def categorize_source_port(df):
    """
    Crea categorías basadas en rangos de puertos para la columna sourcePort.
    
    Parámetros:
    df : pd.DataFrame
        DataFrame con los datos cargados.
        
    Retorna:
    pd.DataFrame : DataFrame con la columna sourcePort categorizada.
    """
    # Definir rangos de puertos
    bins = [0, 1023, 49151, 65535]
    labels = ['Well-Known Ports', 'Registered Ports', 'Dynamic/Private Ports']
    df['sourcePortCategory'] = pd.cut(df['sourcePort'], bins=bins, labels=labels, right=False)
    
    return df


def categorize_ip_by_subnet(ip):
    if ip.startswith('192.168.1.'):
        return '192.168.1.0/24'
    elif ip.startswith('192.168.2.'):
        return '192.168.2.0/24'
    elif ip.startswith('192.168.3.'):
        return '192.168.3.0/24'
    elif ip.startswith('192.168.4.'):
        return '192.168.4.0/24'
    elif ip.startswith('192.168.5.'):
        return '192.168.5.0/24'
    elif ip.startswith('192.168.6.'):
        return '192.168.6.0/24'
    elif ipaddress.ip_address(ip).is_loopback:
        return 'loopback'
    elif ipaddress.ip_address(ip).is_multicast:
        return 'multicast'
    elif ipaddress.ip_address(ip).is_private:
        return 'private (other)'
    elif ipaddress.ip_address(ip).is_link_local:
        return 'link-local'
    elif ipaddress.ip_address(ip) in ipaddress.ip_network('100.64.0.0/10'):
        return 'carrier-grade NAT'
    elif ipaddress.ip_address(ip) in ipaddress.ip_network('192.0.2.0/24') or \
         ipaddress.ip_address(ip) in ipaddress.ip_network('198.51.100.0/24') or \
         ipaddress.ip_address(ip) in ipaddress.ip_network('203.0.113.0/24'):
        return 'documentation'
    else:
        return 'External'

def apply_ip_categorization(df, source_col='source', destination_col='destination'):
    df['source'] = df[source_col].apply(categorize_ip_by_subnet)
    df['destination'] = df[destination_col].apply(categorize_ip_by_subnet)
    return df


def apply_log(df):
    columns = ['totalSourceBytes', 'totalDestinationBytes', 'totalSourcePackets', 'totalDestinationPackets']

    for c in columns:
        df[c] = np.log(df[c])

    return df

