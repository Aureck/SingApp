import os
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump

from constants import SK_MODEL_PATH, CSV_PATH, LABEL_COL, N_FEATS, MIN_CONF
from helpers import ensure_dir, save_meta


def train_from_csv():
    # --- Validación inicial ---
    if not os.path.exists(CSV_PATH) or os.stat(CSV_PATH).st_size == 0:
        raise RuntimeError("No hay CSV con keypoints. Ejecuta create_keypoints.py primero.")

    # Leer dataset completo
    df = pd.read_csv(CSV_PATH).dropna()

    if LABEL_COL not in df.columns:
        raise RuntimeError("CSV sin columna de etiqueta.")

    y = df[LABEL_COL].astype(str).values
    X = df.drop(LABEL_COL, axis=1).astype(np.float32).values

    # Validar número de features
    if X.shape[1] != N_FEATS:
        raise RuntimeError(f"N_FEATS esperado={N_FEATS}, encontrado={X.shape[1]}")

    # Split train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Pipeline
    pipeline = make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            penalty="l2",
            random_state=42,
            class_weight="balanced",
            max_iter=1000,
            tol=1e-3,
            n_jobs=-1
        )
    )

    # Entrenar
    pipeline.fit(X_train, y_train)

    # Validación
    y_pred = pipeline.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, digits=3)

    print("\n[VALIDACIÓN]")
    print(f"Accuracy: {acc*100:.2f}%")
    print(report)

    # Guardar modelo
    ensure_dir(os.path.dirname(SK_MODEL_PATH))
    dump(pipeline, SK_MODEL_PATH)

    # Guardar meta info
    save_meta({
        "trained_on": "keypoints+aug",
        "label_col": LABEL_COL,
        "n_features": N_FEATS,
        "classes": sorted(set(y)),
        "min_conf": MIN_CONF,
        "val_accuracy": acc
    })

    print(f"[MODEL] Guardado en {SK_MODEL_PATH} | clases={len(set(y))}")


if __name__ == "__main__":
    train_from_csv()
