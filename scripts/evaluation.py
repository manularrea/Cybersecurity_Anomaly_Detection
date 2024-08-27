import os
import joblib
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import DataLoader
import numpy as np

def plot_metrics(y_true, y_pred, y_prob, model_name, output_dir, label_encoder):
    # Crear directorios si no existen
    roc_dir = os.path.join(output_dir, 'roc_curves')
    prc_dir = os.path.join(output_dir, 'precision_recall_curves')
    cm_dir = os.path.join(output_dir, 'confusion_matrices')
    os.makedirs(roc_dir, exist_ok=True)
    os.makedirs(prc_dir, exist_ok=True)
    os.makedirs(cm_dir, exist_ok=True)

    # Reporte de clasificación
    report = classification_report(y_true, y_pred)
    print(f"{model_name} Classification Report:\n", report)

    # Resumen de clases en y_true
    unique, counts = np.unique(y_true, return_counts=True)
    print(f"Resumen de clases en y_true para {model_name}:")
    for label, count in zip(label_encoder.classes_, counts):
        print(f"Clase {label}: {count} ejemplos positivos")

    # Curva ROC multiclase
    plt.figure()
    for i, class_label in enumerate(label_encoder.classes_):
        if np.sum(y_true == i) > 0:  # Asegurarse de que haya ejemplos positivos en la clase
            fpr, tpr, _ = roc_curve(y_true == i, y_prob[:, i])
            plt.plot(fpr, tpr, label=f'{model_name} ROC {class_label}')
        else:
            print(f"No positive samples in y_true for class {class_label}. ROC curve not plotted.")

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(roc_dir, f'{model_name}_roc_curve.png'))
    plt.close()

    # Curva Precision-Recall para multiclase
    plt.figure()
    for i, class_label in enumerate(label_encoder.classes_):
        if np.sum(y_true == i) > 0:  # Asegurarse de que haya ejemplos positivos en la clase
            precision, recall, _ = precision_recall_curve(y_true == i, y_prob[:, i])
            plt.plot(recall, precision, label=f'{model_name} Precision-Recall {class_label}')
        else:
            print(f"No positive samples in y_true for class {class_label}. Precision-Recall curve not plotted.")

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} Precision-Recall Curve')
    plt.legend()
    plt.savefig(os.path.join(prc_dir, f'{model_name}_precision_recall_curve.png'))
    plt.close()

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} Confusion Matrix')
    plt.savefig(os.path.join(cm_dir, f'{model_name}_confusion_matrix.png'))
    plt.close()

def evaluate_models(data_file, model_dir, output_dir):
    loader = DataLoader(data_file)
    df = loader.load_data()

    X = df.drop(columns=['Label'])
    y = df['Label']

    # Codificar las etiquetas
    label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
    y_encoded = label_encoder.transform(y)

    models = ["logistic_regression", "random_forest", "xgboost"]

    for model_name in models:
        model = joblib.load(os.path.join(model_dir, f"{model_name}_model.pkl"))
        
        # Predicciones en el conjunto de prueba
        y_pred_test = model.predict(X)
        y_prob_test = model.predict_proba(X)

        # Graficar métricas para el conjunto de prueba
        plot_metrics(y_encoded, y_pred_test, y_prob_test, f"{model_name}_test", output_dir, label_encoder)

if __name__ == "__main__":
    evaluate_models("data/processed/engineered_data.csv", "data/models/", "data/plots/")
