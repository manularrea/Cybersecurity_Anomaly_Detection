import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from cleaner import *

class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, params=None):
        self.params=params

    def fit(self, X, y = None):
        # Ajustes basados en X
        return self
    
    def transform(self, X, y = None):
        # Transformaciones personalizadas
        X_transformed = X.copy()
       
        # Etiquetamos las IP's en su categoría
        X_transformed = apply_ip_categorization(X_transformed)

        # Convertimos las fechas a timestamp
        X_transformed = convert_datetime(X_transformed)

        # Creación de variable cíclica basada en la hora. 
        X_transformed = calculate_temporal_distance(X_transformed)

        # Eliminamos la columna de timestamp
        X_transformed = X_transformed.drop(columns=['startDateTime', 'stopDateTime'])

        #Le aplicamos logaritmo a las variables numéricas para suavizar sus datos
        #X_transformed = apply_log(X_transformed)

        # Transformamos las flags a formato ASCII
        X_transformed = transform_tcp_flags(X_transformed)

        return X_transformed