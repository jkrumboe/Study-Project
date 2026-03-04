# Monitoring Snow and Ice in Iceland -- Testing the EOPF Zarr Format

## Motivation

Iceland's glaciers are shrinking, with seasonal melt-freeze cycles superimposed on a persistent long-term decline. To monitor these changes at scale, we need methods that minimize data movement while preserving scientific fidelity. This project evaluates the new EOPF Zarr format for operational glacier monitoring by implementing two contrasting algorithms and benchmarking them systematically.

## Objectives

1. Implement a **Zarr-optimized, chunk-aware algorithm** that iteratively processes small grid cells and exploits the chunked structure of the Zarr format to load only the data actually needed.
2. Implement a **native-like, full-scene algorithm** that processes entire Sentinel-2 scenes as a baseline comparison.
3. Conduct a **systematic benchmark** across eight grid resolutions (1 km to 25 km), measuring processing time, hardware utilization (RAM, CPU, network, disk I/O), and data quality.

## Method

### Seeds

Glacier seed points are derived from [Iceland's CORINE Land Cover data](https://www.natt.is/en/resources/open-data/data-download). Two seed datasets were prepared in QGIS:
- **203 points**: One centroid per individual glacier polygon classified by CLC.
- **21 points**: One point per cluster of connected or nearby glaciers (within 5 km distance).

### Zarr-Optimized Algorithm

1. **Create Grid**: A regular grid (default 10 km x 10 km) in ISN2004 (EPSG:5325) covering Iceland. Cells containing seed points are marked as initial candidates.
2. **Query STAC**: For each candidate cell, query the [EOPF STAC Catalog](https://stac.core.eopf.eodc.eu) for Sentinel-2 L2A scenes intersecting the cell within the specified time period.
3. **Load Zarr Data**: Load only the relevant chunks from the Zarr store -- specifically B03 (Green, 10 m) and B11 (SWIR1, 20 m) -- and apply the Scene Classification Layer (SCL) to mask clouds, shadows, and water.
4. **Compute Median NDSI**: Resample B11 to 10 m, calculate NDSI = (B03 - B11) / (B03 + B11) per scene, reproject all scenes to a common ISN2004 grid, and compute the temporal median. Pixels with NDSI >= 0.42 are classified as snow/ice.
5. **Spatial Expansion**: If more than 30% of valid pixels in a cell are classified as snow/ice, all four neighboring cells (N, S, E, W) are added as new candidates.
6. **Iterate** until no new candidates remain.

### Native-Like Algorithm

Processes full Sentinel-2 scenes (~110 x 110 km at 10 m resolution) without spatial subsetting. This approach failed on all tested environments due to insufficient RAM, demonstrating the necessity of chunked processing for resource-constrained systems.

### Benchmarking

Each grid resolution was tested with the same seed point and parameters. Hardware metrics (RAM, CPU, network throughput, disk I/O) were collected via the Proxmox VE API. Per-cell timing breakdowns (STAC query, Zarr loading, NDSI computation, tile saving, spatial expansion) were recorded for each run.

## Key Findings

- **Zarr chunking works**: Clear processing time differences across grid sizes confirm that selective chunk access delivers measurable performance benefits. Band-level filtering saves significant resources compared to the SAFE format.
- **Iterative monitoring is effective**: The spatial expansion approach adapts to the actual glacier extent rather than relying on fixed bounding boxes, avoiding unnecessary data processing.
- **Data loading dominates**: STAC queries and Zarr chunk fetching account for a large share of processing time; the actual NDSI computation is comparatively fast.
- **Optimal grid size depends on post-processing complexity**: For lightweight workflows, 15-25 km cells are most efficient; for computationally expensive post-processing, smaller cells reduce wasted area at the cost of more data fetches.
- **Full-scene processing is infeasible on limited hardware**: The native-like approach crashed due to memory constraints, highlighting the advantage of chunk-aware access.

## Project Structure

```
.
├── iceland_ice_monitoring.ipynb   # Main notebook (algorithm implementations, benchmarks, conclusion)
├── glacier_monitoring.py          # Standalone CLI script for production use
├── requirements.txt               # Python dependencies
├── data/                          # Glacier seed GeoJSON files
│   ├── Iceland_Seeds_21.geojson   # 21 clustered seed points
│   └── Iceland_Seeds_203.geojson  # 203 individual seed points
├── output/                        # Benchmark results (per grid size)
├── images/                        # Figures used in the notebook
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Jupyter Notebook

Open `iceland_ice_monitoring.ipynb` and execute the cells sequentially. The notebook walks through each algorithm step with visualizations and explanations.

### Standalone Script

For production or automated workflows, use the CLI script:

```bash
# Process all seeds with default parameters (July 2025, 10 km grid)
python glacier_monitoring.py --seeds data/Iceland_Seeds_203.geojson --output results

# Process a single seed by FID
python glacier_monitoring.py --seeds data/Iceland_Seeds_203.geojson --fid 5 --output results

# Custom time period and thresholds
python glacier_monitoring.py --seeds data/Iceland_Seeds_203.geojson --output results \
  --date-start 2025-06-01 --date-end 2025-09-30 \
  --ndsi-threshold 0.40 --snow-threshold 0.25 --grid-size 15000
```

The script exports timestamped output files to the specified directory:
- `processed_grid_*.geojson` -- Processed cells with snow coverage metadata
- `ndsi_combined_*.tif` -- Combined NDSI raster (GeoTIFF, LZW compressed)
- `snow_mask_*.tif` -- Binary snow/ice mask (GeoTIFF)
- `statistics_*.json` -- Processing statistics and configuration

## Team

- **Robin Gummels** -- Zarr-optimized algorithm, standalone script, Documentation
- **Kian Jay Lenert** -- Seeds
- **Humam Hikmat** -- Native Sentinel algorithm
- **Justin Krumboehmer** -- Benchmarking
