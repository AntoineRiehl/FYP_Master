# Semantic Atlas

**Interactive Exploration of Semantic and Experiential Relationships Across Cultural Domains**

Semantic Atlas is an MSc Data Science project that investigates whether cultural entities can be organised and explored spatially according to different notions of similarity.

The final artefact contains interactive semantic maps for **movies, music, and restaurants**, together with cross-domain representations that allow heterogeneous entities to be explored within a shared space.

The project combines text-based semantic representation, dimensionality reduction, clustering, evaluation, and an interactive React/Vite research interface.

---

## Project Overview

Traditional recommender systems usually reduce large collections to ranked lists of suggested items. Semantic Atlas takes a different approach: instead of only recommending a small number of items, it exposes the wider structure of the item space itself.

Each entity is represented as a point on a two-dimensional atlas. Nearby points are intended to represent semantically or experientially related entities, while regions and local neighbourhoods support exploratory investigation.

The project develops and compares three families of atlases:

### Mono-domain atlases

Independent semantic spaces are constructed for:

- **Movies**
- **Music**
- **Restaurants**

These representations preserve domain-specific descriptive information and are intended for detailed within-domain exploration.

### Cross-domain General Semantic atlases

Shared semantic spaces are constructed for:

- **Movies + Music**
- **Movies + Music + Restaurants**

These atlases use a shared textual representation to investigate whether heterogeneous cultural entities can form meaningful cross-domain semantic neighbourhoods.

### Cross-domain Feel atlases

Shared experiential representations are constructed for:

- **Movies + Music**
- **Movies + Music + Restaurants**

Rather than focusing only on literal semantic content, the Feel representation maps entities using a shared set of experiential dimensions such as valence, activation, warmth, scale, tension, refinement, nostalgia, and wonder.

In total, the final artefact contains **seven atlas configurations**.

---

## Main Technologies

### Data science and modelling

- Python
- pandas
- NumPy
- scikit-learn
- TF-IDF
- Sentence-Transformers
- `sentence-transformers/all-MiniLM-L6-v2`
- cosine similarity
- PCA
- UMAP
- clustering and neighbourhood analysis

### Frontend

- React
- Vite
- JavaScript / TypeScript
- interactive canvas-based rendering
- D3 / Pixi-based visualisation components

---

## Data Sources

The project combines several public cultural datasets.

| Domain | Core data sources | Review / text enrichment |
| --- | --- | --- |
| Movies | MovieLens 32M | IMDb Review Dataset |
| Music | Last.fm, MusicBrainz | CritiqueBrainz |
| Restaurants | Yelp Open Dataset | Yelp reviews and tips |

After preprocessing, the principal entity populations used by the project were approximately:

| Domain | Entities | Entities with reviews |
| --- | ---: | ---: |
| Movies | 84,432 | 42,765 |
| Music | 189,948 | 5,131 |
| Restaurants | 33,941 | 33,941 |

Raw third-party datasets are **not redistributed in this repository**. They should be obtained from their original providers in accordance with their respective licences and terms of use.

---

## Semantic Representation

The modelling pipeline combines lexical and contextual text representations.

### TF-IDF

TF-IDF is used to represent descriptive textual information such as tags, genres, categories, and other entity-level semantic text.

### Sentence-Transformer embeddings

Reviews are encoded using the pretrained:

```text
sentence-transformers/all-MiniLM-L6-v2
```

When multiple reviews are available for an entity, individual review embeddings are mean-pooled into a single review representation.

### Representation fusion

Where both base semantic information and review information are available, the project combines the two representations using a balanced fusion strategy.

For the Feel atlases, semantic and review text are encoded into the same Sentence-Transformer space before being mapped onto a shared 13-dimensional experiential representation.

---

## Projection and Atlas Construction

High-dimensional representations are projected to two dimensions using UMAP.

The main configuration uses:

```text
n_neighbors = 15
min_dist = 0.1
random_state = 42
```

Cosine distance is used for the main semantic spaces, while the standardised Feel representation is projected using Euclidean distance.

The resulting coordinates are enriched with:

- semantic clusters and regions
- nearest neighbours
- region composition
- centrality information
- cross-domain analogues
- within-domain popularity
- experiential Feel profiles

These outputs are exported into frontend-ready atlas bundles consumed by the interactive application.

---

## Interactive Research Tool

The final frontend was implemented in React/Vite after an initial Streamlit prototype.

The interface is designed around a map-like exploration model and supports:

- continuous pan and zoom
- progressive levels of detail
- persistent entity selection
- search and recentering
- nearest semantic neighbours
- closest entities by domain
- cluster and region information
- experiential Feel profiles
- review inspection
- cross-domain analogues
- switching between atlas configurations

The goal is to support exploratory investigation rather than conventional ranked recommendation.

---

## Evaluation

The project evaluates several complementary aspects of atlas quality.

### Representation and fusion evaluation

Alternative fusion weights were compared using neighbourhood coherence measures. A balanced 50/50 fusion was retained as a consistent default because it achieved the strongest or near-strongest performance across the evaluated domains.

### Mono-domain evaluation

The mono-domain atlases were evaluated using:

- neighbourhood label overlap
- lift relative to dataset baselines
- trustworthiness of the 2D projection
- clustering statistics
- qualitative neighbourhood inspection

Headline neighbourhood-coherence lifts included:

- Movies: **1.82× genre lift**
- Movies: **1.95× macro-genre lift**
- Music: **7.03× tag lift**
- Restaurants: **2.75× category lift**

Projection trustworthiness was:

- Movies: **0.6976**
- Music: **0.8216**
- Restaurants: **0.9566**

### Cross-domain evaluation

The General Semantic and Feel approaches were compared using cross-domain neighbourhood mixing and qualitative inspection.

For Movies + Music:

- General Semantic cross-domain neighbour share: **2.20%**
- Feel cross-domain neighbour share: **20.61%**

For Movies + Music + Restaurants:

- General Semantic cross-domain neighbour share: **1.78%**
- Feel cross-domain neighbour share: **15.75%**

The results indicate that the Feel representation produces substantially stronger cross-domain integration, while also creating a more continuous space with weaker discrete cluster separation.

---

## Repository Structure

The repository is organised around four main concerns:

```text
Semantic Atlas
│
├── data/
│   └── local / generated datasets and intermediate files
│
├── src/
│   ├── domain-specific preprocessing
│   ├── feature engineering
│   ├── semantic representations
│   ├── projection and clustering
│   └── atlas-building pipelines
│
├── evaluation/
│   └── representation, neighbourhood and atlas evaluation
│
└── frontend/
    └── React / Vite interactive research application
```

The exact structure may contain additional scripts, notebooks, evaluation outputs, and utility modules used during development.

---

## Generated Atlas Data

The final precomputed atlas JSON files are **not committed directly to the Git repository** because several exceed GitHub's standard per-file size limit.

The complete modelling pipelines required to generate these files are included in the repository.

The final application expects the generated atlas bundles inside:

```text
frontend/public/data/
```

A packaged copy of the final generated atlas data can be distributed separately, for example through the repository's **Releases** section.

Once downloaded, the data bundle should be extracted so that the expected directory structure is restored under:

```text
frontend/public/data/
```

---

## Running the Frontend

### Requirements

Install a recent version of:

- Node.js
- npm

Then open a terminal in the repository and run:

```bash
cd frontend
npm install
npm run dev
```

Vite will display the local development URL in the terminal.

> **Important:** the application requires the generated atlas data to be present under `frontend/public/data/`.

---

## Running the Modelling Pipeline

The modelling pipeline is implemented in Python.

A typical environment should include the main dependencies used by the project, including:

```text
pandas
numpy
scikit-learn
sentence-transformers
umap-learn
```

Install the project's Python dependencies using the dependency file provided in the repository where available, for example:

```bash
pip install -r requirements.txt
```

The project contains separate preprocessing and atlas-building components for movies, music, and restaurants, together with cross-domain and Feel pipelines.

Because the original raw datasets are not redistributed, reproducing the complete pipeline requires downloading the datasets listed in the **Data Sources** section first.

---

## Development Process

The project was developed iteratively.

Early mono-domain atlases were first explored through Streamlit/Plotly prototypes. These prototypes were useful for validating the modelling approach but exposed limitations in navigation, rendering, zoom behaviour, persistent interaction, and large-scale exploratory use.

The research-tool requirements were subsequently refined and the final interface was rebuilt using React/Vite with a dedicated atlas-rendering architecture.

The modelling process also evolved from independent mono-domain atlases to shared General Semantic representations and finally to the cross-domain Feel representation.

---

## Academic Context

This repository accompanies the MSc Data Science final project:

**Semantic Atlas: Interactive Exploration of Semantic and Experiential Relationships Across Cultural Domains**

The project was completed at the **University of the West of England (UWE Bristol)** in 2026.

The repository is intended to provide:

- source code for the modelling pipeline
- evaluation scripts and outputs
- frontend source code
- implementation and reproducibility material supporting the submitted report

---

## Limitations

Several limitations should be considered when interpreting the project:

- review coverage differs substantially between domains
- the Feel dimensions and semantic anchors are engineered modelling choices
- two-dimensional UMAP projections necessarily introduce some distortion
- discrete clustering may be less suitable for continuous experiential spaces
- cross-domain experiential similarity does not have an objective ground-truth label set
- the final frontend was functionally evaluated but not subjected to a controlled user study

---

## Future Work

Possible extensions include:

- formal user studies comparing atlas-based exploration with conventional recommendation interfaces
- refinement and validation of the Feel dimensions
- alternative continuous visualisations for experiential spaces
- additional cultural domains
- personalised atlas views
- improved large-scale delivery of generated atlas data
- packaged desktop distribution of the final application

---

## Author

**Antoine Riehl**  
MSc Data Science, UWE Bristol  
2026
