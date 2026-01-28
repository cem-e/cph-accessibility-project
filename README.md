# Accessibility and inequality in Copenhagen

This repository contains the code and selected outputs for the MSc thesis:

Accessibility and Inequality in Copenhagen: Investigating the Relationship Between Income and Urban Opportunities  
Cem Ergin, IT University of Copenhagen, January 2026

The project examines whether differences in pedestrian accessibility to urban amenities are associated with income differences across Copenhagen districts.

## Project motivation

Urban accessibility is often assumed to be evenly distributed in Copenhagen due to its compact urban form and strong public infrastructure. This thesis investigates whether that assumption holds when accessibility is measured using realistic walking distances on the street network and compared across income levels.

The focus is on everyday pedestrian accessibility rather than transport travel times or straight-line distances.

---

## Research question

To what extent is district-level pedestrian accessibility to essential and recreational amenities associated with income across Copenhagen districts?

---

## Method overview

Accessibility is calculated using a network-based approach:

- A pedestrian street network is constructed from OpenStreetMap
- Walking-time distances are computed using shortest paths on the network
- Accessibility is measured as distance to the nearest amenity in eight categories:
  - Food and drink
  - Supermarkets
  - Education
  - Health
  - Parks
  - Playgrounds
  - Public transport stops
  - Cultural amenities

Accessibility values are aggregated to the district level and compared with district-level income statistics.

To avoid bias from uninhabited areas such as harbours and industrial zones, a population-based filtering step is applied. Street network nodes located in very low-population areas are excluded from aggregation, producing accessibility measures that better reflect residents’ lived experience.

---

## Data Pipeline
The analysis is implemented as a three-step data processing pipeline. Each step produces explicit outputs that are used by the next stage.

### Step 1: Node-level accessibility computation

A pedestrian street network is built for Copenhagen and Frederiksberg using OpenStreetMap data. Accessibility is computed at the level of individual street network nodes.

For each node, walking-time distance is calculated to the nearest amenity in each category using network shortest paths. Nodes are also annotated with local population values derived from a population raster, allowing identification of nodes located in very low-population areas.

This step produces node-level accessibility metrics and GeoJSON outputs used for visualisation.

### Step 2: District-level aggregation

Node-level accessibility values are spatially joined to district polygons and aggregated per district.

For each district, median, mean, and upper-percentile accessibility values are computed across nodes. Isolation-aware versions of these metrics are calculated by excluding nodes in low-population areas.

This step produces a district-level accessibility table used for statistical comparison.

### Step 3: Income integration

District-level accessibility metrics are merged with median income data.

Because district naming conventions differ between datasets, a deterministic name mapping is used to ensure correct alignment. The final output combines accessibility metrics, income values, and district geometry for analysis and web-based visualisation.

---

## Repository structure

pipeline/  
Contains the Python-based data processing and accessibility computation pipeline.

web/  
Contains files used for interactive visualisation of accessibility and income patterns.

---

## Key findings (summary)

Overall pedestrian accessibility in Copenhagen is high across districts.

Median income shows little association with general accessibility to everyday amenities.

Stronger differences appear for leisure-oriented amenities, particularly cultural facilities, which are more unevenly distributed and concentrated near the city centre.

Distance from the city centre plays a larger role than income in explaining accessibility differences.

---

## Data and reproducibility

Large raw datasets and intermediate outputs are not included in this repository to keep it lightweight.

The repository focuses on the computational methods, analysis logic, and visual outputs used in the thesis.

---

## Academic context

This project was developed as part of an MSc thesis in Software Design at the IT University of Copenhagen.
