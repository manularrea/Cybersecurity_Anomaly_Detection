import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        return pd.read_csv(self.file_path)

class DataSaver:
    def __init__(self, output_path):
        self.output_path = output_path

    def save_data(self, df):
        df.to_csv(self.output_path, index=False)

class Scaler:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, X):
        return self.scaler.fit_transform(X)

    def transform(self, X):
        return self.scaler.transform(X)

    def save(self, path):
        import joblib
        joblib.dump(self.scaler, path)

    def load(self, path):
        import joblib
        self.scaler = joblib.load(path)

