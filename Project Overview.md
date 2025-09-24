# Praktikum Datenanalyse - Benutzerhandbuch

Willkommen zu eurem Praktikumstag! Ihr werdet heute drei spannende Datensätze erforschen und dabei lernen, wie man Daten analysiert und visualisiert.

## Setup und Installation

### 1. Repository klonen
```bash
git clone https://github.com/CatraMyBeloved/praktikum_aufgaben
cd praktikum_aufgaben
```

### 2. Abhängigkeiten installieren
```bash
pip install pandas matplotlib seaborn numpy jupyter
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
import pandas as pd
from helpers import *

# Datensatz laden (wählt einen aus!)
movies = pd.read_csv('filtered_movie_metadata.csv')
olympics = pd.read_csv('filtered_athlete_events.csv') 
spotify = pd.read_csv('filtered_spotify_data.csv')
```

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

## Hilfe und Tipps

### Bei Problemen:
1. **Fehlermeldungen lesen** - oft steht da genau, was falsch ist
2. **Spaltennamen prüfen** - `data.columns` zeigt alle verfügbaren Spalten
3. **Beispiele aus diesem Handbuch** kopieren und anpassen
4. **Fragen stellen** - wir helfen gerne!

### GenAI nutzen:
Ihr könnt immer Gemini um Hilfe bitten. Wenn ihr das korrekte Model auswählt, hat es automatisch wissen über diese Aufgaben.

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

Viel Erfolg beim Erkunden der Daten! 🚀