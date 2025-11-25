# Monitoring Snow and Ice in Iceland

## Motivation
Iceland’s glaciers are shrinking, with seasonal melt–freeze cycles superimposed on a persistent long‑term decline (see: Icelandic glacier loss - AntarcticGlaciers.org and A Revised Snow Cover Algorithm to Improve Discrimination between Snow and Clouds: A Case Study in Gran Paradiso National Park). To monitor these changes at scale, we need methods that minimize data movement while preserving scientific fidelity. This project leverages the new EOPF Zarr format to exploit chunked, cloud‑native reads for efficient snow and ice mapping from Sentinel‑2.

## Objectives
1. Efficient, reproducible computation of monthly glacier and snow area in Iceland. 
2. Efficient storage and I/O by prioritizing only the pixels likely to contain snow/ice. 
3. A head‑to‑head comparison between a chunked tile‑aware algorithm and a baseline that processes complete Sentinel scenes, evaluating runtime, I/O volume and result parity.

## Methods
Seeds are glacier polygons from CORINE Land Cover (CLC). For each seed we define a bespoke 10×10 km tile. For every month, we load all sub‑scenes intersecting the tile (via Zarr chunks), build a median composite, compute a per‑pixel Normalized Difference Snow Index (NDSI), and classify snow/ice using NDSI > 0.42. If the tile’s snow cover exceeds 30%, we iteratively load neighboring tiles to capture contiguous snowfields.

Comparative algorithm: always load full Sentinel‑2 scenes, compute NDSI, classify with the same threshold, and if a scene’s snow cover exceeds 30%, process all adjacent and overlapping scenes. For both pipelines, we compute, per month and seed, the area with NDSI > 0.42 and aggregate to glacier and national levels. We then generate a time series of the fractional area and absolute area.

## Expected results
We will benchmark runtime, bytes read and compare area estimates between methods. We expect to see strong intra-annual variability (winter maxima, summer minima) and a multi‑year decline in snow/ice extent. The tile‑aware EOPF Zarr approach should reduce I/O and storage and deliver shorter runtimes while preserving agreement with scene‑level results.

## Team members and work distribution
The exact work distribution is a little bit unsure, because it’s hard to determine the complexity of developing this two algorithms, but we are estimating the following distribution:
- Robin Gummels: Zarr algorithm
- Kian Jay Lenert: Seeds, documentation/introduction, benchmark Zarr algorithm
- Humam Hikmat: Native sentinel algorithm
- Justin Krumböhmer: Benchmark native sentinel algorithm, temporal analysis of glacier development
