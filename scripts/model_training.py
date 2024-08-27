import os
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.preprocessing import LabelEncoder
import joblib
import pandas as pd
from utils import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def train_model(input_file, model_output_dir):
    loader = DataLoader(input_file)
    df = loader.load_data()

    # Separate features and labels
    X = df.drop(columns=['Label'])
    y = df['Label']

    # Encode the labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Print distribution before undersampling
    print("Distribución antes de Undersampling:", pd.Series(y_encoded).value_counts().values)

    # Apply undersampling to balance the classes
    undersampler = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = undersampler.fit_resample(X, y_encoded)

    # Print distribution after undersampling
    print("Distribución después de Undersampling:", pd.Series(y_resampled).value_counts().values)

    # Use StratifiedShuffleSplit to split the data in a stratified manner
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in sss.split(X_resampled, y_resampled):
        X_train, X_test = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
        y_train, y_test = y_resampled[train_index], y_resampled[test_index]

    #                  Dataframes for explainability graphs
    #================================TEST================================#

    # Convert y_test to a DataFrame and reset its index to ensure correct alignment
    y_test_df = pd.DataFrame(y_test, columns=['Label'], index=X_test.index)

    # Concatenate X_test and y_test_df to create the final test_data DataFrame
    test_data = pd.concat([X_test, y_test_df], axis=1)

    # Save the test data for future analysis
    test_data.to_csv("data/processed/engineered_test_data.csv", index=False)

    #================================TEST================================#

    #================================TRAIN================================#

    # Convert y_test to a DataFrame and reset its index to ensure correct alignment
    y_train_df = pd.DataFrame(y_train, columns=['Label'], index=X_train.index)

    # Concatenate X_test and y_test_df to create the final test_data DataFrame
    train_data = pd.concat([X_train, y_train_df], axis=1)

    # Save the test data for future analysis
    train_data.to_csv("data/processed/engineered_train_data.csv", index=False)

    #================================TRAIN================================#


    # Definición de modelos con optimización de parámetros
    models = {
        "logistic_regression": LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200, class_weight='balanced'),
        "random_forest": RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1, class_weight='balanced'),
        "xgboost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', scale_pos_weight=len(y_resampled) / sum(y_resampled == 1)),
        "svm": SVC(kernel='rbf', class_weight='balanced', probability=True),
        "mlp": MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=300, random_state=42)


    }

    # Entrenamiento y guardado de modelos
    for name, model in models.items():
        # Crear un pipeline que escale y entrene
        pipeline = Pipeline([
            #('scaler', StandardScaler()),
            ('model', model)
        ])
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, os.path.join(model_output_dir, f"{name}_model.pkl"))

        # Validación cruzada
        scores = cross_val_score(pipeline, X_train, y_train, cv=5)
        print(f"{name} Cross-validation scores: {scores.mean():.3f} +/- {scores.std():.3f}")

    # Guardar el label encoder para el futuro
    joblib.dump(label_encoder, os.path.join(model_output_dir, "label_encoder.pkl"))

if __name__ == "__main__":
    train_model("data/processed/engineered_data.csv", "data/models/")
