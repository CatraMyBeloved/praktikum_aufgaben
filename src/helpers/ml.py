# helpers/ml.py
"""
Machine Learning Helfer-Modul für Klassifikation.
Fokus: Spotify Genre-Vorhersage mit Decision Trees.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)


# =============================================================================
# Internal Validation Helpers
# =============================================================================

def _validate_dataframe(data: pd.DataFrame) -> None:
    """Prüft ob der DataFrame nicht leer ist."""
    if data.empty:
        raise ValueError(
            "Fehler: Der DataFrame ist leer (0 Zeilen).\n"
            "  Problem: Machine Learning benötigt Daten zum Trainieren.\n"
            "  Lösung: Prüfen Sie, ob die Daten korrekt geladen wurden oder ob ein Filter zu streng war.\n"
            "  Tipp: Verwenden Sie 'len(data)' um die Anzahl der Zeilen zu prüfen."
        )


def _validate_target(y: pd.Series, target_name: str) -> None:
    """Prüft ob die Zielvariable mindestens 2 Klassen hat."""
    n_classes = y.nunique()
    if n_classes < 2:
        classes = y.unique().tolist()
        raise ValueError(
            f"Fehler: Zielvariable '{target_name}' hat nur {n_classes} Klasse(n): {classes}.\n"
            f"  Problem: Klassifikation benötigt mindestens 2 verschiedene Klassen.\n"
            f"  Mögliche Ursachen:\n"
            f"    1. Der DataFrame wurde zu stark gefiltert\n"
            f"    2. Die falsche Spalte wurde als Ziel gewählt\n"
            f"  Lösung: Prüfen Sie die Zielspalte mit: data['{target_name}'].value_counts()"
        )


def _warn_zero_variance(X: pd.DataFrame) -> None:
    """Warnt bei Features mit nur einem eindeutigen Wert."""
    for col in X.columns:
        if X[col].nunique() <= 1:
            unique_val = X[col].iloc[0] if len(X) > 0 else "N/A"
            print(
                f"Hinweis: Feature '{col}' hat nur einen eindeutigen Wert ({unique_val}).\n"
                f"  Problem: Diese Spalte trägt keine Information zur Vorhersage bei.\n"
                f"  Empfehlung: Diese Spalte aus den Features entfernen."
            )

__all__ = [
    'prepare_ml_data',
    'split_data',
    'train_model',
    'evaluate_model',
    'predict',
    'plot_confusion_matrix',
    'plot_feature_importance',
    'plot_decision_tree'
]


def prepare_ml_data(
    data: pd.DataFrame,
    target: str,
    features: Optional[List[str]] = None,
    task: str = "classification",
    encode_categorical: str = "onehot",
    max_cardinality: int = 20,
    drop_first: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Trennt die Daten in Features (X) und Zielvariable (y).

    Parameters
    ----------
    data : pd.DataFrame
        Der Datensatz mit allen Spalten.
    target : str
        Name der Zielvariable (z.B. 'track_genre').
    features : List[str], optional
        Liste der Feature-Spalten. Wenn None, werden alle numerischen Spalten verwendet.
    task : str, default="classification"
        Art der Aufgabe:
        - "classification" - Prüft ob Ziel mind. 2 Klassen hat
        - "regression" - Prüft ob Ziel numerisch ist
    encode_categorical : str, default="onehot"
        Wie kategoriale Features kodiert werden sollen:
        - "onehot": One-Hot Encoding (erstellt Dummy-Variablen)
        - "label": Label Encoding (0, 1, 2, ...)
        - "none": Keine Kodierung (Fehler bei kategorialen Features)
    max_cardinality : int, default=20
        Warnt bei kategorialen Features mit mehr eindeutigen Werten.
    drop_first : bool, default=True
        Bei One-Hot Encoding: Erste Kategorie weglassen (vermeidet Multikollinearität).

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        X (Features) und y (Zielvariable).

    Example
    -------
    >>> # Klassifikation
    >>> X, y = prepare_ml_data(spotify, target='track_genre',
    ...                        features=['danceability', 'energy', 'tempo'])
    >>> # Regression
    >>> X, y = prepare_ml_data(movies, target='revenue',
    ...                        features=['budget', 'runtime'], task='regression')
    """
    # 1. Validate empty DataFrame
    _validate_dataframe(data)

    # 2. Validate target exists
    if target not in data.columns:
        available = ', '.join(data.columns[:10].tolist())
        if len(data.columns) > 10:
            available += f', ... (+{len(data.columns) - 10} weitere)'
        raise ValueError(
            f"Fehler: Zielspalte '{target}' nicht gefunden.\n"
            f"  Verfügbare Spalten: {available}\n"
            f"  Tipp: Achten Sie auf Groß-/Kleinschreibung."
        )

    # 3. Validate target based on task
    y = data[target].copy()

    if task == "regression":
        # For regression: target should be numeric
        if not np.issubdtype(y.dtype, np.number):
            raise ValueError(
                f"Fehler: Zielvariable '{target}' ist nicht numerisch (Typ: {y.dtype}).\n"
                f"  Problem: Regression benötigt eine numerische Zielvariable.\n"
                f"  Lösung: Verwenden Sie task='classification' für kategoriale Ziele\n"
                f"          oder wählen Sie eine numerische Spalte als Ziel."
            )
    else:
        # For classification: target should have 2+ classes
        _validate_target(y, target)

    # 4. Select features
    if features is None:
        # Automatisch alle numerischen Spalten auswählen (außer Zielvariable)
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        features = [col for col in numeric_cols if col != target]

        if len(features) == 0:
            all_dtypes = data.dtypes.value_counts().to_dict()
            raise ValueError(
                f"Fehler: Keine numerischen Features gefunden.\n"
                f"  Spaltentypen im DataFrame: {all_dtypes}\n"
                f"  Lösung: Geben Sie Features explizit an und nutzen Sie encode_categorical='onehot'\n"
                f"  Beispiel: prepare_ml_data(data, '{target}', features=['spalte1', 'spalte2'], encode_categorical='onehot')"
            )

        print(f"Automatisch {len(features)} numerische Features ausgewählt: {', '.join(features[:5])}" +
              (f", ... (+{len(features)-5} weitere)" if len(features) > 5 else ""))
    else:
        # Prüfen ob alle angegebenen Features existieren
        missing = [f for f in features if f not in data.columns]
        if missing:
            available = ', '.join(data.columns[:10].tolist())
            if len(data.columns) > 10:
                available += f', ... (+{len(data.columns) - 10} weitere)'
            raise ValueError(
                f"Fehler: Feature(s) nicht gefunden: {', '.join(missing)}\n"
                f"  Verfügbare Spalten: {available}\n"
                f"  Tipp: Achten Sie auf Groß-/Kleinschreibung."
            )

    # 5. Extract features
    X = data[features].copy()

    # 6. Identify categorical columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # 7. Handle categorical encoding
    if cat_cols:
        if encode_categorical == "none":
            raise ValueError(
                f"Fehler: Kategoriale Features gefunden: {cat_cols}\n"
                f"  Problem: ML-Modelle können nicht direkt mit Text arbeiten.\n"
                f"  Lösung: Verwenden Sie encode_categorical='onehot' oder 'label'\n"
                f"  Beispiel: prepare_ml_data(data, '{target}', features={features}, encode_categorical='onehot')"
            )

        # Warn about high cardinality
        for col in cat_cols:
            n_unique = X[col].nunique()
            if n_unique > max_cardinality:
                print(
                    f"Warnung: Feature '{col}' hat {n_unique} eindeutige Werte (> {max_cardinality}).\n"
                    f"  Problem: Hohe Kardinalität kann zu Overfitting führen.\n"
                    f"  Empfehlung: Verwenden Sie encode_categorical='label' oder entfernen Sie diese Spalte."
                )

        # Handle missing values in categorical columns BEFORE encoding
        for col in cat_cols:
            if X[col].isna().any():
                mode_val = X[col].mode()
                if len(mode_val) > 0:
                    X[col] = X[col].fillna(mode_val.iloc[0])
                    print(f"Hinweis: Fehlende Werte in '{col}' mit Modus aufgefüllt.")

        if encode_categorical == "onehot":
            X = pd.get_dummies(X, columns=cat_cols, drop_first=drop_first)
            print(f"One-Hot Encoding angewendet auf: {cat_cols} ({len(X.columns)} Features nach Encoding)")

        elif encode_categorical == "label":
            for col in cat_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
            print(f"Label Encoding angewendet auf: {cat_cols}")

    # 8. Handle missing values in numeric columns
    numeric_cols_in_X = X.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols_in_X:
        na_mask = X[numeric_cols_in_X].isna()
        if na_mask.any().any():
            na_count = na_mask.sum().sum()
            X[numeric_cols_in_X] = X[numeric_cols_in_X].fillna(X[numeric_cols_in_X].median())
            print(f"Hinweis: {na_count} fehlende Werte in numerischen Features mit Median aufgefüllt.")

    # 9. Warn about zero-variance features
    _warn_zero_variance(X)

    print(f"Daten vorbereitet: {len(X)} Datensätze, {len(X.columns)} Features, Ziel: '{target}'")

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    task: str = "classification"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Teilt die Daten in Trainings- und Testdaten auf.

    Warum ist das wichtig?
    - Wir trainieren das Modell nur mit den Trainingsdaten
    - Die Testdaten simulieren "neue, ungesehene" Daten
    - So können wir prüfen, ob das Modell wirklich gelernt hat

    Parameters
    ----------
    X : pd.DataFrame
        Features.
    y : pd.Series
        Zielvariable.
    test_size : float, default=0.2
        Anteil der Testdaten (0.2 = 20%).
    random_state : int, default=42
        Zufallszahl für Reproduzierbarkeit.
    task : str, default="classification"
        Art der Aufgabe:
        - "classification" - Stratifiziertes Splitting (gleiche Klassenverteilung)
        - "regression" - Zufälliges Splitting

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test

    Example
    -------
    >>> X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    >>> # Für Regression:
    >>> X_train, X_test, y_train, y_test = split_data(X, y, task='regression')
    """
    if task == "regression":
        # No stratification for regression
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    else:
        # Stratified split for classification
        # Validate minimum samples per class for stratification
        class_counts = y.value_counts()
        min_per_class = class_counts.min()
        smallest_class = class_counts.idxmin()

        if min_per_class < 2:
            raise ValueError(
                f"Fehler: Klasse '{smallest_class}' hat nur {min_per_class} Datensatz.\n"
                f"  Problem: Stratifiziertes Splitting benötigt mindestens 2 Samples pro Klasse.\n"
                f"  Klassenverteilung:\n" +
                "\n".join(f"    - '{cls}': {count}" for cls, count in class_counts.items()) +
                f"\n  Lösung: Entfernen Sie seltene Klassen oder fügen Sie mehr Daten hinzu."
            )

        # Check if we have enough samples for the split
        min_needed = int(np.ceil(1 / test_size))
        if min_per_class < min_needed:
            print(
                f"Warnung: Klasse '{smallest_class}' hat nur {min_per_class} Samples.\n"
                f"  Bei test_size={test_size} werden mindestens {min_needed} Samples pro Klasse empfohlen.\n"
                f"  Das Splitting wird trotzdem versucht."
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

    train_pct = int((1 - test_size) * 100)
    test_pct = int(test_size * 100)

    print(f"Daten aufgeteilt: Training: {len(X_train)} ({train_pct}%), Test: {len(X_test)} ({test_pct}%)")

    return X_train, X_test, y_train, y_test


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "decision_tree",
    task: str = "classification",
    max_depth: Optional[int] = 5,
    n_neighbors: int = 5
) -> Any:
    """
    Trainiert ein Klassifikations- oder Regressionsmodell.

    Parameters
    ----------
    X_train : pd.DataFrame
        Trainings-Features.
    y_train : pd.Series
        Trainings-Zielvariable.
    model_type : str, default="decision_tree"
        Art des Modells:
        - "decision_tree" - Einfacher Entscheidungsbaum
        - "random_forest" - Mehrere Bäume stimmen ab
        - "knn" - K-Nearest Neighbors (findet ähnliche Datenpunkte)
        - "svm" - Support Vector Machine
        - "gradient_boosting" - Boosted Trees (fortgeschritten)
    task : str, default="classification"
        Art der Aufgabe:
        - "classification" - Vorhersage von Kategorien (z.B. Genre)
        - "regression" - Vorhersage von Zahlen (z.B. Preis, Bewertung)
    max_depth : int, optional, default=5
        Maximale Tiefe für Baum-Modelle. Begrenzt Overfitting.
    n_neighbors : int, default=5
        Anzahl Nachbarn für KNN.

    Returns
    -------
    Any
        Das trainierte Modell.

    Example
    -------
    >>> # Klassifikation (Standard)
    >>> model = train_model(X_train, y_train, model_type='decision_tree')
    >>> # Regression
    >>> model = train_model(X_train, y_train, model_type='decision_tree', task='regression')
    """
    valid_types = ["decision_tree", "random_forest", "knn", "svm", "gradient_boosting"]
    valid_tasks = ["classification", "regression"]

    if task not in valid_tasks:
        raise ValueError(
            f"Fehler: Unbekannte Aufgabe '{task}'.\n"
            f"  Gültige Optionen:\n"
            f"    - 'classification': Vorhersage von Kategorien (z.B. Genre, Klasse)\n"
            f"    - 'regression': Vorhersage von Zahlenwerten (z.B. Preis, Bewertung)"
        )

    is_regression = (task == "regression")
    task_label = "Regression" if is_regression else "Klassifikation"

    if model_type == "decision_tree":
        if is_regression:
            model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        else:
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        model_name = "Decision Tree"
        param_info = f"max_depth={max_depth}" if max_depth else "unbegrenzte Tiefe"

    elif model_type == "random_forest":
        if is_regression:
            model = RandomForestRegressor(max_depth=max_depth, n_estimators=100, random_state=42)
        else:
            model = RandomForestClassifier(max_depth=max_depth, n_estimators=100, random_state=42)
        model_name = "Random Forest"
        param_info = f"max_depth={max_depth}, 100 Bäume"

    elif model_type == "knn":
        if is_regression:
            model = KNeighborsRegressor(n_neighbors=n_neighbors)
        else:
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model_name = "K-Nearest Neighbors"
        param_info = f"n_neighbors={n_neighbors}"

    elif model_type == "svm":
        if is_regression:
            model = SVR(kernel='rbf')
        else:
            model = SVC(kernel='rbf', random_state=42)
        model_name = "Support Vector Machine"
        param_info = "kernel=rbf"

    elif model_type == "gradient_boosting":
        if is_regression:
            model = GradientBoostingRegressor(
                max_depth=max_depth if max_depth else 3,
                n_estimators=100,
                learning_rate=0.1,
                random_state=42
            )
        else:
            model = GradientBoostingClassifier(
                max_depth=max_depth if max_depth else 3,
                n_estimators=100,
                learning_rate=0.1,
                random_state=42
            )
        model_name = "Gradient Boosting"
        param_info = f"max_depth={max_depth if max_depth else 3}, 100 Bäume"

    else:
        raise ValueError(
            f"Fehler: Unbekannter Modelltyp '{model_type}'.\n"
            f"  Gültige Optionen:\n"
            f"    - 'decision_tree': Einfacher Entscheidungsbaum (gut zum Lernen)\n"
            f"    - 'random_forest': Mehrere Bäume stimmen ab (oft genauer)\n"
            f"    - 'knn': Findet ähnliche Datenpunkte\n"
            f"    - 'svm': Findet optimale Trennebenen\n"
            f"    - 'gradient_boosting': Baut Bäume sequenziell auf"
        )

    model.fit(X_train, y_train)
    print(f"Modell trainiert: {model_name} für {task_label} ({param_info})")

    return model


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task: str = "classification",
    show_details: bool = True
) -> Dict[str, Any]:
    """
    Bewertet das Modell auf den Testdaten.

    Parameters
    ----------
    model : Any
        Das trainierte Modell.
    X_test : pd.DataFrame
        Test-Features.
    y_test : pd.Series
        Test-Zielvariable (wahre Werte).
    task : str, default="classification"
        Art der Aufgabe:
        - "classification" - Klassifikationsmetriken (Accuracy, Confusion Matrix)
        - "regression" - Regressionsmetriken (MSE, RMSE, MAE, R²)
    show_details : bool, default=True
        Ob detaillierte Metriken angezeigt werden sollen.

    Returns
    -------
    Dict[str, Any]
        Für Klassifikation: 'accuracy', 'predictions', 'confusion_matrix'
        Für Regression: 'r2', 'mse', 'rmse', 'mae', 'predictions'

    Example
    -------
    >>> # Klassifikation
    >>> results = evaluate_model(model, X_test, y_test)
    >>> print(f"Genauigkeit: {results['accuracy']:.1%}")
    >>> # Regression
    >>> results = evaluate_model(model, X_test, y_test, task='regression')
    >>> print(f"R²: {results['r2']:.3f}")
    """
    predictions = model.predict(X_test)

    if task == "regression":
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print(f"R² Score: {r2:.3f} (1.0 = perfekt, 0.0 = so gut wie Durchschnitt)")
        print(f"RMSE: {rmse:.3f} (durchschnittlicher Fehler in Originaleinheit)")

        if show_details:
            print(f"\nDetaillierte Metriken:")
            print(f"  MSE (Mean Squared Error): {mse:.3f}")
            print(f"  RMSE (Root MSE): {rmse:.3f}")
            print(f"  MAE (Mean Absolute Error): {mae:.3f}")
            print(f"  R² (Bestimmtheitsmaß): {r2:.3f}")

            # Show prediction range
            print(f"\nVorhersage-Statistik:")
            print(f"  Tatsächlich: Min={y_test.min():.2f}, Max={y_test.max():.2f}, Mean={y_test.mean():.2f}")
            print(f"  Vorhergesagt: Min={predictions.min():.2f}, Max={predictions.max():.2f}, Mean={predictions.mean():.2f}")

        return {
            'r2': r2,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'predictions': pd.Series(predictions, index=y_test.index)
        }

    else:  # classification
        acc = accuracy_score(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)

        print(f"Genauigkeit: {acc:.1%} ({acc:.1%} der Datensätze wurden richtig klassifiziert)")

        if show_details:
            print("\nKlassifikationsbericht:")
            print(classification_report(y_test, predictions))

        return {
            'accuracy': acc,
            'predictions': pd.Series(predictions, index=y_test.index),
            'confusion_matrix': cm
        }


def predict(
    model: Any,
    data: pd.DataFrame,
    features: Optional[List[str]] = None
) -> pd.Series:
    """
    Macht Vorhersagen für neue Daten.

    Parameters
    ----------
    model : Any
        Das trainierte Modell.
    data : pd.DataFrame
        Neue Daten für Vorhersagen.
    features : List[str], optional
        Liste der Feature-Spalten. Wenn None, werden alle Spalten verwendet.

    Returns
    -------
    pd.Series
        Vorhergesagte Klassen.

    Example
    -------
    >>> neue_songs = spotify.sample(5)
    >>> vorhersagen = predict(model, neue_songs, features=['danceability', 'energy', 'tempo'])
    """
    if features is not None:
        # Check if all features exist
        missing = [f for f in features if f not in data.columns]
        if missing:
            raise ValueError(
                f"Fehler: Feature(s) nicht gefunden: {', '.join(missing)}\n"
                f"  Verfügbare Spalten: {', '.join(data.columns[:10].tolist())}"
                + (f", ... (+{len(data.columns) - 10} weitere)" if len(data.columns) > 10 else "")
            )
        X = data[features]
    else:
        X = data

    # Validate feature count matches model expectation
    if hasattr(model, 'n_features_in_'):
        expected = model.n_features_in_
        actual = X.shape[1]
        if actual != expected:
            raise ValueError(
                f"Fehler: Falsche Anzahl Features.\n"
                f"  Erwartet: {expected} Features (vom Training)\n"
                f"  Erhalten: {actual} Features\n"
                f"  Lösung: Verwenden Sie dieselben Features wie beim Training.\n"
                f"  Tipp: Speichern Sie die Feature-Liste beim Training: feature_list = X_train.columns.tolist()"
            )

    predictions = model.predict(X)

    print(f"Vorhersagen gemacht für {len(data)} Datensätze.")

    return pd.Series(predictions, index=data.index)


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: pd.Series,
    title: Optional[str] = None,
    top_n: int = 10
) -> None:
    """
    Visualisiert die Konfusionsmatrix.

    Die Konfusionsmatrix zeigt, welche Klassen wie oft verwechselt wurden.
    - Diagonale: Korrekte Vorhersagen
    - Außerhalb der Diagonale: Fehler

    Parameters
    ----------
    y_true : pd.Series
        Wahre Werte.
    y_pred : pd.Series
        Vorhergesagte Werte.
    title : str, optional
        Titel des Plots.
    top_n : int, default=10
        Maximale Anzahl der angezeigten Klassen.

    Example
    -------
    >>> plot_confusion_matrix(y_test, results['predictions'])
    """
    # Klassen ermitteln
    classes = sorted(y_true.unique())

    # Wenn zu viele Klassen, nur die häufigsten nehmen
    if len(classes) > top_n:
        top_classes = y_true.value_counts().head(top_n).index.tolist()
        mask = y_true.isin(top_classes)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        classes = sorted(top_classes)
        print(f"Hinweis: Zeige nur die {top_n} häufigsten Klassen.")

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    # Achsenbeschriftungen
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='Wahrer Wert',
           xlabel='Vorhersage')

    # Labels rotieren
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Zahlen in die Zellen schreiben
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    if title:
        ax.set_title(title)
    else:
        ax.set_title('Konfusionsmatrix')

    fig.tight_layout()
    plt.show()


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 10,
    title: Optional[str] = None
) -> pd.DataFrame:
    """
    Zeigt die Wichtigkeit der Features für das Modell.

    Feature Importance zeigt, welche Variablen am meisten zur
    Vorhersage beitragen.

    Parameters
    ----------
    model : Any
        Das trainierte Modell (muss feature_importances_ haben).
    feature_names : List[str]
        Namen der Features.
    top_n : int, default=10
        Anzahl der wichtigsten Features, die angezeigt werden.
    title : str, optional
        Titel des Plots.

    Returns
    -------
    pd.DataFrame
        DataFrame mit Feature-Wichtigkeiten.

    Example
    -------
    >>> importance_df = plot_feature_importance(model, X_train.columns)
    """
    if not hasattr(model, 'feature_importances_'):
        print("Hinweis: Dieses Modell unterstützt keine Feature-Wichtigkeit.")
        print("Verwende Decision Tree, Random Forest oder Gradient Boosting für diese Funktion.")
        return pd.DataFrame({'Feature': feature_names, 'Wichtigkeit': [None] * len(feature_names)})

    # DataFrame mit Wichtigkeiten erstellen
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Wichtigkeit': model.feature_importances_
    }).sort_values('Wichtigkeit', ascending=False)

    # Top N Features für Plot
    plot_df = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(plot_df)), plot_df['Wichtigkeit'].values, color='steelblue')
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df['Feature'].values)
    ax.invert_yaxis()  # Wichtigste oben
    ax.set_xlabel('Wichtigkeit')

    if title:
        ax.set_title(title)
    else:
        ax.set_title('Feature-Wichtigkeit')

    # Werte an Balken schreiben
    for bar, val in zip(bars, plot_df['Wichtigkeit'].values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center')

    plt.tight_layout()
    plt.show()

    print(f"\nTop {min(top_n, len(importance_df))} wichtigste Features:")
    for i, row in plot_df.iterrows():
        print(f"  {row['Feature']}: {row['Wichtigkeit']:.3f}")

    return importance_df


def plot_decision_tree(
    model: Any,
    feature_names: List[str],
    class_names: Optional[List[str]] = None,
    max_depth: int = 3
) -> None:
    """
    Visualisiert die Struktur eines Decision Trees.

    Der Plot zeigt, wie das Modell Entscheidungen trifft:
    - Jeder Knoten zeigt eine Bedingung (z.B. "danceability <= 0.5")
    - Blätter zeigen die vorhergesagte Klasse

    Parameters
    ----------
    model : Any
        Das trainierte Decision Tree Modell.
    feature_names : List[str]
        Namen der Features.
    class_names : List[str], optional
        Namen der Klassen. Wenn None, werden sie aus dem Modell gelesen.
    max_depth : int, default=3
        Maximale Tiefe für die Visualisierung.

    Example
    -------
    >>> plot_decision_tree(model, X_train.columns)
    """
    if not isinstance(model, (DecisionTreeClassifier, DecisionTreeRegressor)):
        print("Hinweis: plot_decision_tree funktioniert nur mit Decision Trees.")
        print("Für andere Modelle verwende plot_feature_importance() stattdessen.")
        return

    is_classifier = isinstance(model, DecisionTreeClassifier)

    if class_names is None:
        if is_classifier:
            class_names = [str(c) for c in model.classes_]
        else:
            class_names = None  # Regression trees don't have classes

    fig, ax = plt.subplots(figsize=(20, 10))

    plot_tree(model,
              feature_names=feature_names,
              class_names=class_names,
              filled=True,
              rounded=True,
              max_depth=max_depth,
              ax=ax,
              fontsize=10)

    plt.title(f'Decision Tree (Tiefe: {max_depth})')
    plt.tight_layout()
    plt.show()

    actual_depth = model.get_depth()
    print(f"Baum-Visualisierung (zeigt Tiefe {max_depth} von {actual_depth} Ebenen)")
