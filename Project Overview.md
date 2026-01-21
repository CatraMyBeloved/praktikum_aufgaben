# Praktikum Datenanalyse - Benutzerhandbuch

Willkommen zu eurem Praktikumstag! Ihr werdet heute drei Datensätze erforschen und dabei lernen, wie man Daten analysiert und visualisiert.

## Setup und Installation

### 1. Repository klonen
```bash
git clone https://github.com/CatraMyBeloved/praktikum_aufgaben
cd praktikum_aufgaben
```

### 2. Abhängigkeiten installieren
```bash
pip install pandas matplotlib seaborn numpy jupyter scikit-learn
```

### 3. Jupyter Notebook starten
```bash
jupyter notebook
```

## Die Datensätze

Ihr habt die Wahl zwischen drei verschiedenen Datensätzen:

### 🎬 Filme (IMDB)
- **Datei**: `filtered_movie_metadata.csv`
- **Größe**: ~3.800 Filme
- **Interessante Spalten**:
  - `movie_title` - Filmtitel
  - `imdb_score` - Bewertung (1-10)
  - `title_year` - Erscheinungsjahr
  - `genres` - Genres (z.B. "Action|Adventure|Sci-Fi")
  - `director_name` - Regisseur
  - `budget` - Budget in Dollar
  - `gross` - Einnahmen in Dollar
  - `duration` - Länge in Minuten

### 🏅 Olympische Spiele
- **Datei**: `filtered_athlete_events.csv`
- **Größe**: ~14.000 Medaillengewinner (Sommer-Olympia seit 1988)
- **Interessante Spalten**:
  - `Name` - Athletenname
  - `NOC` - Land (3-Buchstaben-Code)
  - `Sport` - Sportart
  - `Event` - Spezifische Disziplin
  - `Medal` - Medaille (Gold/Silver/Bronze)
  - `Year` - Olympisches Jahr
  - `Age` - Alter des Athleten
  - `Height`, `Weight` - Körpermaße

### 🎵 Spotify Songs
- **Datei**: `filtered_spotify_data.csv`
- **Größe**: ~18.000 populäre Songs
- **Interessante Spalten**:
  - `track_name` - Songname
  - `artists` - Künstler
  - `popularity` - Popularität (0-100)
  - `track_genre` - Genre
  - `danceability` - Tanzbarkeit (0-1)
  - `energy` - Energie (0-1)
  - `valence` - Positivität (0-1)
  - `tempo` - Geschwindigkeit (BPM)

## Helper-Funktionen

Wir haben für euch eine Sammlung von Hilfsfunktionen erstellt, die komplizierte pandas-Syntax verstecken:

### Daten laden und Helper importieren
```python
from helpers import load_dataset, load_noc_regions

movies = load_dataset('movies')       # Filtered version by default
olympics = load_dataset('olympics')   # Enthält nur relevante Medaillen-Daten
spotify = load_dataset('spotify')     # Beliebte Tracks mit Audio-Features
noc_regions = load_noc_regions()      # Länderzuordnung für Olympia
```

- `load_dataset(name, filtered=True)` lädt automatisch Dateien aus dem `data/` Ordner.
- Setze `filtered=False`, wenn du die unbearbeiteten Originaldaten brauchst.
- Mit `load_noc_regions()` bekommst du die Referenztabelle für Olympia-Länder.

### 1. `filter_data()` - Daten filtern

Filtert Datensätze nach verschiedenen Kriterien:

```python
# Beispiele für Filme
filter_data(movies, title_year=2010)  # Nur Filme von 2010
filter_data(movies, imdb_score_min=8.0)  # Nur sehr gut bewertete Filme
filter_data(movies, genres='Action')  # Filme mit "Action" im Genre
filter_data(movies, title_year_min=2000, imdb_score_min=7.0)  # Kombiniert

# Beispiele für Olympia
filter_data(olympics, NOC='GER')  # Nur deutsche Athleten
filter_data(olympics, Sport='Swimming')  # Nur Schwimmen
filter_data(olympics, Year_min=2008)  # Ab 2008

# Beispiele für Spotify
filter_data(spotify, popularity_min=80)  # Nur sehr populäre Songs
filter_data(spotify, track_genre='pop')  # Nur Pop-Songs
filter_data(spotify, danceability_min=0.8)  # Sehr tanzbare Songs
```

**Tipps:**
- Für Mindest-/Höchstwerte: `spalte_min=X` oder `spalte_max=X`
- Für Text: wird automatisch gesucht (enthält Text)
- Für Listen: `spalte=[wert1, wert2, wert3]`

### 2. `add_columns()` - Neue Spalten erstellen

Erstellt neue Spalten mit einfachen Formeln:

```python
# Beispiele für Filme
movies_extended = add_columns(movies,
    decade="title_year // 10 * 10",  # Jahrzehnt (2000, 2010, etc.)
    is_long="duration > 120",        # Ist der Film länger als 2 Stunden?
    profit="gross - budget",         # Gewinn berechnen
    score_category="'Gut' if imdb_score >= 7 else 'Mittelmäßig'"
)

# Beispiele für Olympia  
olympics_extended = add_columns(olympics,
    is_germany="NOC == 'GER'",      # Deutsche Athleten?
    bmi="Weight / (Height/100)**2",  # BMI berechnen
    age_group="'Jung' if Age < 25 else 'Älter'"
)

# Beispiele für Spotify
spotify_extended = add_columns(spotify,
    is_danceable="danceability > 0.7",
    energy_level="'Hoch' if energy > 0.7 else 'Niedrig'",
    popularity_score="popularity / 10"  # Auf Skala 0-10
)
```

### 3. `summarize_by_group()` - Gruppierungen und Zusammenfassungen

Fasst Daten nach Kategorien zusammen:

```python
# Durchschnittliche Bewertung nach Genre
summarize_by_group(movies, 'genres', 'imdb_score', 'mean')

# Medaillen pro Land zählen
summarize_by_group(olympics, 'NOC', 'Medal', 'count') 

# Durchschnittliche Tanzbarkeit nach Genre
summarize_by_group(spotify, 'track_genre', 'danceability', 'mean')

# Mehrere Spalten gleichzeitig
summarize_by_group(movies, 'director_name', ['imdb_score', 'gross'], 'mean')
```

**Verfügbare Funktionen:** `'mean'`, `'count'`, `'sum'`, `'min'`, `'max'`

### 4. `get_top_n()` - Top/Flop Listen

Findet die besten oder schlechtesten Einträge:

```python
# Top 10 beste Filme
get_top_n(movies, 'imdb_score', n=10)

# Top 5 schlechteste Filme  
get_top_n(movies, 'imdb_score', n=5, direction='bottom')

# Länder mit den meisten Medaillen (mit Spaltenauswahl)
get_top_n(olympics, 'Medal')['NOC', 'Name', 'Sport', 'Medal']

# Populärste Songs
get_top_n(spotify, 'popularity', n=15)
```

### 5. `plot()` - Visualisierungen erstellen

Erstellt schnell verschiedene Diagramme:

```python
# Verteilungen anzeigen
plot(movies, 'distribution', x='imdb_score')
plot(olympics, 'distribution', x='Sport') 
plot(spotify, 'distribution', x='danceability')

# Zusammenhänge zwischen zwei Variablen
plot(movies, 'relationship', x='budget', y='gross')
plot(olympics, 'relationship', x='Height', y='Weight')  
plot(spotify, 'relationship', x='danceability', y='popularity')

# Mit Trendlinie
plot(movies, 'relationship', x='budget', y='gross', add_trend=True)

# Gruppiert nach Kategorien
plot(movies, 'distribution', x='imdb_score', group_by='genres')
plot(olympics, 'relationship', x='Height', y='Weight', group_by='Sport')
```

## Machine Learning Helper-Funktionen

Für den ML-Teil gibt es zusätzliche Funktionen, um Klassifikationsmodelle zu trainieren:

```python
from helpers.ml import *
```

### 1. `prepare_ml_data()` - Features und Zielvariable trennen

```python
# Features (X) und Zielvariable (y) vorbereiten
X, y = prepare_ml_data(spotify, target='track_genre',
                       features=['danceability', 'energy', 'tempo', 'valence'])
```

### 2. `split_data()` - Trainings- und Testdaten aufteilen

```python
# 80% Training, 20% Test
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
```

### 3. `train_model()` - Modell trainieren

```python
# Decision Tree trainieren
model = train_model(X_train, y_train, model_type='decision_tree', max_depth=5)

# Oder Random Forest
model = train_model(X_train, y_train, model_type='random_forest', max_depth=5)
```

### 4. `evaluate_model()` - Modell bewerten

```python
# Genauigkeit und Details anzeigen
results = evaluate_model(model, X_test, y_test)
print(f"Genauigkeit: {results['accuracy']:.1%}")
```

### 5. `predict()` - Vorhersagen machen

```python
# Genre für neue Songs vorhersagen
vorhersagen = predict(model, neue_songs, features=['danceability', 'energy', 'tempo', 'valence'])
```

### 6. Visualisierungen

```python
# Konfusionsmatrix - welche Klassen werden verwechselt?
plot_confusion_matrix(y_test, results['predictions'])

# Feature-Wichtigkeit - welche Features sind entscheidend?
plot_feature_importance(model, feature_names)

# Decision Tree visualisieren
plot_decision_tree(model, feature_names, max_depth=3)
```

## Erste Schritte

### 1. Datensatz wählen und laden
```python
import pandas as pd
from helpers import *

# Wählt euren Lieblingsdatensatz!
data = pd.read_csv('filtered_movie_metadata.csv')  # oder olympics/spotify

# Ersten Eindruck gewinnen
print(data.shape)  # Wie viele Zeilen und Spalten?
print(data.columns)  # Welche Spalten gibt es?
data.head()  # Erste 5 Zeilen anschauen
```

### 2. Grundlegende Erkundung
```python
# Verteilungen anschauen
plot(data, 'distribution', x='spaltenname')

# Zusammenfassende Statistiken
data.describe()

# Fehlende Werte prüfen
data.isnull().sum()
```

### 3. Interessante Fragen stellen
Siehe die Beispielfragen weiter unten!

## Beispielhafte Forschungsfragen

### Für Filme:
- Welche Genres haben die höchsten Bewertungen?
- Werden längere Filme besser bewertet?
- Welche Regisseure sind am erfolgreichsten?
- Führen höhere Budgets zu höheren Einnahmen?
- Wie haben sich Filmebewertungen über die Jahrzehnte entwickelt?

### Für Olympia:
- Welche Länder dominieren im Schwimmen/Leichtathletik?
- Wie hat sich die Medaillenverteilung seit 1988 verändert?
- In welchen Sportarten sind die Athleten am ältesten/jüngsten?
- Gibt es einen Zusammenhang zwischen Körpergröße und Sportart?

### Für Spotify:
- Was macht einen Song populär?
- Sind tanzbarere Songs beliebter?
- Welche Audio-Features hängen zusammen?
- Haben verschiedene Genres charakteristische "Signaturen"?
- Wie unterscheiden sich die Genres in Energie und Positivität?

### Für Machine Learning:
- Kann man das Genre eines Songs anhand der Audio-Features vorhersagen?
- Welche Features sind am wichtigsten für die Genre-Vorhersage?
- Welche Genres werden am häufigsten verwechselt?
- Ist ein Random Forest besser als ein einzelner Decision Tree?
- Wie verändert sich die Genauigkeit mit mehr oder weniger Genres?

## Hilfe und Tipps

### Bei Problemen:
1. **Fehlermeldungen lesen** - oft steht da genau, was falsch ist
2. **Spaltennamen prüfen** - `data.columns` zeigt alle verfügbaren Spalten
3. **Beispiele aus diesem Handbuch** kopieren und anpassen
4. **Fragen stellen** - wir helfen gerne!

### GenAI nutzen:
Ihr könnt immer Gemini um Hilfe bitten. Wenn ihr das korrekte Model auswählt, hat es automatisch wissen über diese Aufgaben.gi

**Gute Prompts:**
- "Ich bekomme einen KeyError bei Spalte 'Genre' - hier ist mein Code: ..."
- "Erkläre mir, was dieser pandas-Befehl macht: data.groupby('Sport').count()"  
- "Wie kann ich in Python zwei Spalten multiplizieren?"

**Schlechte Prompts:**
- "Fix meinen Code"  
- "Mach eine Analyse"

### Nützliche pandas-Tricks:
```python
# Spalten auswählen
data[['spalte1', 'spalte2', 'spalte3']]

# Informationen über Datensatz
data.info()
data.describe()

# Einzigartige Werte sehen
data['spaltenname'].unique()
data['spaltenname'].value_counts()
```

## Für Fortgeschrittene: Pandas & Sklearn direkt nutzen

Wer tiefer einsteigen möchte, kann die gleichen Aufgaben auch direkt mit pandas und sklearn lösen. Hier sind die Entsprechungen zu unseren Helper-Funktionen:

### Pandas: Daten filtern

```python
# Mit Helper:
filter_data(movies, title_year=2010, imdb_score_min=7.0)

# Mit pandas:
movies[(movies['title_year'] == 2010) & (movies['imdb_score'] >= 7.0)]

# Mehrere Werte (isin)
movies[movies['genres'].isin(['Action', 'Comedy', 'Drama'])]

# Text enthält
movies[movies['genres'].str.contains('Action', na=False)]
```

### Pandas: Neue Spalten erstellen

```python
# Mit Helper:
add_columns(movies, profit="gross - budget")

# Mit pandas:
movies['profit'] = movies['gross'] - movies['budget']

# Bedingte Spalten mit np.where
import numpy as np
movies['is_good'] = np.where(movies['imdb_score'] >= 7, 'Gut', 'Schlecht')

# Komplexere Bedingungen mit apply
movies['category'] = movies['imdb_score'].apply(
    lambda x: 'Super' if x >= 8 else ('Gut' if x >= 6 else 'Mäßig')
)
```

### Pandas: Gruppieren und Aggregieren

```python
# Mit Helper:
summarize_by_group(movies, 'genres', 'imdb_score', 'mean')

# Mit pandas:
movies.groupby('genres')['imdb_score'].mean()

# Mehrere Aggregationen
movies.groupby('genres').agg({
    'imdb_score': 'mean',
    'gross': 'sum',
    'movie_title': 'count'
})
```

### Pandas: Sortieren und Top N

```python
# Mit Helper:
get_top_n(movies, 'imdb_score', n=10)

# Mit pandas:
movies.nlargest(10, 'imdb_score')
movies.nsmallest(10, 'imdb_score')  # Bottom 10

# Sortieren
movies.sort_values('imdb_score', ascending=False).head(10)
```

### Pandas: Visualisierungen mit Matplotlib/Seaborn

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Histogramm
plt.hist(movies['imdb_score'], bins=20)
plt.xlabel('IMDB Score')
plt.title('Verteilung der Bewertungen')
plt.show()

# Scatterplot
plt.scatter(movies['budget'], movies['gross'], alpha=0.5)
plt.xlabel('Budget')
plt.ylabel('Einnahmen')
plt.show()

# Seaborn für schönere Plots
sns.histplot(data=movies, x='imdb_score', hue='genres')
sns.scatterplot(data=movies, x='budget', y='gross', hue='genres')
```

### Sklearn: Machine Learning Pipeline

```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Features und Zielvariable
feature_cols = ['danceability', 'energy', 'tempo', 'valence']
X = spotify[feature_cols]
y = spotify['track_genre']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Modell trainieren
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Vorhersagen und Bewertung
y_pred = model.predict(X_test)
print(f"Genauigkeit: {accuracy_score(y_test, y_pred):.1%}")
print(classification_report(y_test, y_pred))

# Konfusionsmatrix
cm = confusion_matrix(y_test, y_pred)

# Feature Importance
importance = pd.DataFrame({
    'Feature': feature_cols,
    'Wichtigkeit': model.feature_importances_
}).sort_values('Wichtigkeit', ascending=False)
```

### Weiterführende Ressourcen

- [Pandas Dokumentation](https://pandas.pydata.org/docs/)
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)
- [Sklearn User Guide](https://scikit-learn.org/stable/user_guide.html)

## Notebooks

| Notebook | Beschreibung |
|----------|-------------|
| `work_environment.ipynb` | Arbeitsumgebung für Datenanalyse |
| `ml_work_environment.ipynb` | Arbeitsumgebung für Machine Learning |
| `spotify_analysis_example.ipynb` | Beispielanalyse mit Spotify-Daten |
| `ml_showcase.ipynb` | Beispiel für Machine Learning mit Erklärungen |

Viel Erfolg beim Erkunden der Daten! 🚀