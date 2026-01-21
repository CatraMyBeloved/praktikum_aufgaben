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
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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
    features: Optional[List[str]] = None
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

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        X (Features) und y (Zielvariable).

    Example
    -------
    >>> X, y = prepare_ml_data(spotify, target='track_genre',
    ...                        features=['danceability', 'energy', 'tempo'])
    """
    # Prüfen ob Zielvariable existiert
    if target not in data.columns:
        available = ', '.join(data.columns[:10].tolist())
        if len(data.columns) > 10:
            available += ', ...'
        raise ValueError(f"Fehler: Spalte '{target}' nicht gefunden. Verfügbare Spalten: {available}")

    # Features auswählen
    if features is None:
        # Automatisch alle numerischen Spalten auswählen (außer Zielvariable)
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        features = [col for col in numeric_cols if col != target]

        if len(features) == 0:
            raise ValueError("Fehler: Keine numerischen Features gefunden.")

        print(f"Automatisch {len(features)} numerische Features ausgewählt: {', '.join(features[:5])}" +
              (f", ... (+{len(features)-5} weitere)" if len(features) > 5 else ""))
    else:
        # Prüfen ob alle angegebenen Features existieren
        missing = [f for f in features if f not in data.columns]
        if missing:
            raise ValueError(f"Fehler: Feature(s) nicht gefunden: {', '.join(missing)}")

    # Features und Zielvariable extrahieren
    X = data[features].copy()
    y = data[target].copy()

    # Fehlende Werte behandeln
    if X.isna().any().any():
        na_count = X.isna().sum().sum()
        X = X.fillna(X.median())
        print(f"Hinweis: {na_count} fehlende Werte in Features mit Median aufgefüllt.")

    print(f"Daten vorbereitet: {len(X)} Datensätze, {len(features)} Features, Ziel: '{target}'")

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
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

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test

    Example
    -------
    >>> X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    """
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
    max_depth: Optional[int] = 5
) -> Any:
    """
    Trainiert ein Klassifikationsmodell.

    Parameters
    ----------
    X_train : pd.DataFrame
        Trainings-Features.
    y_train : pd.Series
        Trainings-Zielvariable.
    model_type : str, default="decision_tree"
        Art des Modells: "decision_tree" oder "random_forest".
    max_depth : int, optional, default=5
        Maximale Tiefe des Baums. Begrenzt Overfitting.

    Returns
    -------
    Any
        Das trainierte Modell.

    Example
    -------
    >>> model = train_model(X_train, y_train, model_type='decision_tree', max_depth=5)
    """
    if model_type == "decision_tree":
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        model_name = "Decision Tree"
    elif model_type == "random_forest":
        model = RandomForestClassifier(max_depth=max_depth, n_estimators=100, random_state=42)
        model_name = "Random Forest"
    else:
        raise ValueError(f"Fehler: model_type muss 'decision_tree' oder 'random_forest' sein, nicht '{model_type}'.")

    model.fit(X_train, y_train)

    depth_info = f"max_depth={max_depth}" if max_depth else "unbegrenzte Tiefe"
    print(f"Modell trainiert: {model_name} ({depth_info})")

    return model


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
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
    show_details : bool, default=True
        Ob detaillierte Metriken angezeigt werden sollen.

    Returns
    -------
    Dict[str, Any]
        Dictionary mit 'accuracy', 'predictions', 'confusion_matrix'.

    Example
    -------
    >>> results = evaluate_model(model, X_test, y_test)
    >>> print(f"Genauigkeit: {results['accuracy']:.1%}")
    """
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    print(f"Genauigkeit: {acc:.1%} ({acc:.1%} der Songs wurden richtig klassifiziert)")

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
        X = data[features]
    else:
        X = data

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
        raise ValueError("Fehler: Das Modell hat keine feature_importances_. "
                        "Verwende ein Decision Tree oder Random Forest Modell.")

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
    if not isinstance(model, DecisionTreeClassifier):
        print("Hinweis: plot_decision_tree funktioniert nur mit Decision Trees.")
        print("Für Random Forest verwende plot_feature_importance() stattdessen.")
        return

    if class_names is None:
        class_names = [str(c) for c in model.classes_]

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
