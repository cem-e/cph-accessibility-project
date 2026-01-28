# Accessibility and inequality in Copenhagen

This repository contains the code and selected outputs for the MSc thesis:

Accessibility and Inequality in Copenhagen: Investigating the Relationship Between Income and Urban Opportunities  
Cem Ergin, IT University of Copenhagen, January 2026

The project examines whether differences in pedestrian accessibility to urban amenities are associated with income differences across Copenhagen districts.

---

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
