#!/usr/bin/env python3
"""
Glacier Monitoring Script using Sentinel-2 L2A Data

This script implements a Zarr-optimized glacier monitoring algorithm with spatial expansion.
It processes Sentinel-2 L2A scenes from the EOPF STAC catalog to detect snow and ice coverage
across Iceland using the NDSI (Normalized Difference Snow Index).

Algorithm Overview:
1. Create a 10km x 10km grid covering Iceland
2. Mark cells containing glacier seed points as initial candidates
3. For each candidate cell:
   - Query STAC catalog for Sentinel-2 L2A scenes
   - Load only relevant Zarr data chunks
   - Compute median NDSI composite
   - Apply quality masking (remove clouds, shadows, water)
4. Apply spatial expansion: Add neighboring cells if snow coverage exceeds threshold
5. Iterate until no new candidates are added
6. Combine all results into unified snow/ice mask
7. Export results to output directory

Author: GitHub Copilot
Date: January 2026
"""

import os
import sys
import gc
import argparse
import warnings
import json
from datetime import datetime
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# Geospatial libraries
import geopandas as gpd
from shapely.geometry import box, Point
from pyproj import Transformer

# Data processing
import xarray as xr
import numpy as np
import pandas as pd
import rioxarray
from rasterio.transform import from_bounds
from rasterio.enums import Resampling

# STAC API for Sentinel-2 data
from pystac_client import Client


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    'epsg_iceland': 5325,  # ISN2004 / Lambert 2004 (Iceland's national projection)
    'bounding_box': [1400000, 100000, 2000000, 500000],  # [minx, miny, maxx, maxy] in ISN2004
    'grid_size': 10000,  # Grid cell size in meters (10 km)
    'ndsi_threshold': 0.42,  # Snow/ice classification threshold
    'snow_percentage_threshold': 0.30,  # Spatial expansion threshold (30% coverage)
    'stac_url': "https://stac.core.eopf.eodc.eu",  # EOPF STAC Catalog endpoint
    'date_start': "2025-07-01",  # Start date (YYYY-MM-DD)
    'date_end': "2025-07-31",  # End date (YYYY-MM-DD)
    'max_iterations': 50,  # Maximum iterations for spatial expansion
    'max_scenes': 20,  # Maximum scenes to process per cell (reduced from 100 for memory)
    'low_memory': True  # Memory-efficient mode: write results to disk incrementally
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_grid(bounds, grid_size, epsg_code):
    """
    Create a regular grid covering the specified bounding box.
    
    Parameters:
    -----------
    bounds : list
        [minx, miny, maxx, maxy] in meters (ISN2004 coordinates)
    grid_size : int
        Size of a grid cell in meters
    epsg_code : int
        EPSG code for the coordinate reference system
        
    Returns:
    --------
    gpd.GeoDataFrame : Grid with cell_id and geometry columns
    """
    xmin, ymin, xmax, ymax = bounds
    
    # Round grid boundaries to multiples of grid_size for clean alignment
    xmin = np.floor(xmin / grid_size) * grid_size
    ymin = np.floor(ymin / grid_size) * grid_size
    xmax = np.ceil(xmax / grid_size) * grid_size
    ymax = np.ceil(ymax / grid_size) * grid_size
    
    # Create grid cells by iterating over x and y coordinates
    grid_cells = []
    grid_ids = []
    cell_id = 0
    
    y = ymin
    while y < ymax:
        x = xmin
        while x < xmax:
            # Create a square polygon for each grid cell
            cell = box(x, y, x + grid_size, y + grid_size)
            grid_cells.append(cell)
            grid_ids.append(cell_id)
            cell_id += 1
            x += grid_size
        y += grid_size
    
    # Create GeoDataFrame with ISN2004 CRS
    grid = gpd.GeoDataFrame({
        'cell_id': grid_ids,
        'geometry': grid_cells
    }, crs=f"EPSG:{epsg_code}")
    
    return grid


def mark_candidate_cells(grid, seeds):
    """
    Mark grid cells that contain at least one seed point as candidates for processing.
    
    Uses spatial join to identify which grid cells intersect with seed points.
    
    Parameters:
    -----------
    grid : gpd.GeoDataFrame
        GeoDataFrame with grid cells
    seeds : gpd.GeoDataFrame
        GeoDataFrame with seed points (glacier locations)
        
    Returns:
    --------
    gpd.GeoDataFrame : Grid with additional columns:
        - is_candidate: bool indicating if the cell contains at least one seed
        - seed_count: number of seeds contained in the cell
    """
    # Initialize all cells as non-candidates
    grid['is_candidate'] = False
    grid['seed_count'] = 0
    
    # Spatial join: identify which grid cells contain which seeds
    joined = gpd.sjoin(grid, seeds, how='inner', predicate='contains')
    
    # Count seeds per grid cell
    if len(joined) > 0:
        candidate_counts = joined.groupby('cell_id').size()
        
        # Mark cells containing seeds as candidates
        grid.loc[grid['cell_id'].isin(candidate_counts.index), 'is_candidate'] = True
        grid.loc[grid['cell_id'].isin(candidate_counts.index), 'seed_count'] = \
            grid.loc[grid['cell_id'].isin(candidate_counts.index), 'cell_id'].map(candidate_counts)
    
    return grid


def query_stac_for_cell(cell, date_start, date_end, epsg_code, stac_url, verbose=False):
    """
    Query EOPF STAC Catalog for Sentinel-2 L2A scenes covering a grid cell.
    
    This function searches for all Sentinel-2 L2A scenes that intersect with
    the given cell's geometry during the specified time period. It handles
    coordinate transformation from ISN2004 to WGS84 (required by STAC API).
    
    Parameters:
    -----------
    cell : pandas.Series or dict
        Cell with 'geometry' attribute (Shapely polygon in ISN2004)
    date_start : str
        Start date in format "YYYY-MM-DD"
    date_end : str
        End date in format "YYYY-MM-DD"
    epsg_code : int
        EPSG code of input cell geometry
    stac_url : str
        STAC catalog URL
    verbose : bool
        If True, print progress information
    
    Returns:
    --------
    list : List of STAC items (pystac.Item objects) with Zarr data URLs
    """
    # Transform cell bounds from ISN2004 to WGS84 (required by STAC API)
    cell_gdf = gpd.GeoDataFrame([cell], crs=f"EPSG:{epsg_code}")
    cell_wgs84 = cell_gdf.to_crs(epsg=4326)
    bbox_wgs84 = cell_wgs84.total_bounds
    
    if verbose:
        print(f"  Querying STAC for cell {cell.get('cell_id', 'unknown')}: {date_start} to {date_end}")
    
    # Connect to EOPF STAC Catalog
    catalog = Client.open(stac_url)
    
    # Search for Sentinel-2 L2A scenes that intersect the cell
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox_wgs84,
        datetime=[date_start, date_end]
    )
    
    # Collect results
    items = list(search.items())
    
    if verbose:
        print(f"    Found {len(items)} scenes")
    
    return items


def load_zarr_data_for_cell(zarr_url, cell_bounds_isn2004, cell_epsg_isn2004):
    """
    Load and extract Zarr data for a specific cell using ESA best practices.
    
    This function handles coordinate transformation from ISN2004 to the product's
    UTM zone, loads only relevant data chunks, and applies quality masking.
    
    Parameters:
    -----------
    zarr_url : str
        URL to the EOPF Sentinel-2 L2A Zarr store
    cell_bounds_isn2004 : tuple
        Cell bounds in ISN2004 coordinates (minx, miny, maxx, maxy)
    cell_epsg_isn2004 : int
        EPSG code of input coordinates
    
    Returns:
    --------
    dict or None : Dictionary containing:
        - 'b03': Green band (10m resolution) as xr.DataArray
        - 'b11': SWIR1 band (20m resolution) as xr.DataArray
        - 'valid_mask': Quality mask (20m resolution)
        - 'metadata': STAC discovery metadata
    """
    try:
        # Open Zarr store as DataTree (ESA recommended approach)
        dt = xr.open_datatree(zarr_url, engine='zarr', chunks={})
        
        # Extract metadata
        metadata = dt.attrs.get('stac_discovery', {})
        product_epsg = metadata.get('properties', {}).get('proj:epsg')
        
        # Transform coordinates from ISN2004 to product UTM zone
        if product_epsg != cell_epsg_isn2004:
            transformer = Transformer.from_crs(cell_epsg_isn2004, product_epsg, always_xy=True)
            minx_in, miny_in, maxx_in, maxy_in = cell_bounds_isn2004
            (minx_utm, miny_utm) = transformer.transform(minx_in, miny_in)
            (maxx_utm, maxy_utm) = transformer.transform(maxx_in, maxy_in)
            minx_utm, maxx_utm = min(minx_utm, maxx_utm), max(minx_utm, maxx_utm)
            miny_utm, maxy_utm = min(miny_utm, maxy_utm), max(miny_utm, maxy_utm)
        else:
            minx_utm, miny_utm, maxx_utm, maxy_utm = cell_bounds_isn2004
        
        # Load bands and quality mask
        b03 = dt.measurements.reflectance.r10m.b03
        b11 = dt.measurements.reflectance.r20m.b11
        scl = dt.conditions.mask.l2a_classification.r20m.scl
        
        # Create quality mask (exclude invalid pixels)
        # SCL values: 0=NoData, 1=Saturated, 3=CloudShadow, 6=Water, 7-9=Clouds
        valid_mask = ~scl.isin([0, 1, 3, 6, 7, 8, 9])
        
        # Clip using coordinate filtering
        x_mask_b03 = (b03.x >= minx_utm) & (b03.x <= maxx_utm)
        y_mask_b03 = (b03.y >= miny_utm) & (b03.y <= maxy_utm)
        b03_clipped = b03.sel(x=b03.x[x_mask_b03], y=b03.y[y_mask_b03])
        
        x_mask_b11 = (b11.x >= minx_utm) & (b11.x <= maxx_utm)
        y_mask_b11 = (b11.y >= miny_utm) & (b11.y <= maxy_utm)
        b11_clipped = b11.sel(x=b11.x[x_mask_b11], y=b11.y[y_mask_b11])
        
        x_mask_scl = (scl.x >= minx_utm) & (scl.x <= maxx_utm)
        y_mask_scl = (scl.y >= miny_utm) & (scl.y <= maxy_utm)
        valid_clipped = valid_mask.sel(x=scl.x[x_mask_scl], y=scl.y[y_mask_scl])
        
        # Apply quality mask to B11
        b11_clipped = b11_clipped.where(valid_clipped)
        
        # Check if data is empty
        if b03_clipped.size == 0 or b11_clipped.size == 0:
            return None
        
        return {
            'b03': b03_clipped,
            'b11': b11_clipped,
            'valid_mask': valid_clipped,
            'metadata': metadata
        }
        
    except Exception as e:
        print(f"    Error loading {zarr_url.split('/')[-1]}: {e}")
        return None


def compute_median_ndsi_for_cell(stac_items, cell_bounds, epsg_code, ndsi_threshold, max_scenes):
    """
    Compute median NDSI composite for a cell from multiple Sentinel-2 scenes.
    
    This function processes all available scenes from all tiles by:
    1. Loading B03 (Green) and B11 (SWIR1) bands
    2. Resampling B11 from 20m to 10m resolution
    3. Computing NDSI = (B03 - B11) / (B03 + B11)
    4. Reprojecting all NDSI arrays to a common ISN2004 grid
    5. Computing temporal median to reduce noise and cloud contamination
    
    Parameters:
    -----------
    stac_items : list
        List of STAC items (Sentinel-2 L2A scenes)
    cell_bounds : tuple
        Cell bounds in ISN2004 coordinates (minx, miny, maxx, maxy)
    epsg_code : int
        EPSG code of input coordinates
    ndsi_threshold : float
        NDSI threshold for snow/ice classification
    max_scenes : int
        Maximum number of scenes to process
    
    Returns:
    --------
    dict or None : Dictionary containing:
        - 'ndsi_median': Median NDSI as xr.DataArray
        - 'snow_mask': Boolean mask where NDSI >= threshold
        - 'snow_percentage': Percentage of snow/ice pixels (based on valid pixels)
        - 'scene_count': Number of scenes successfully loaded
    """
    ndsi_list = []
    scenes_loaded = 0
    
    # Load and process each scene
    for idx, item in enumerate(stac_items[:max_scenes]):
        zarr_url = item.assets["product"].href
        
        # Load Zarr data (handles coordinate transformation automatically)
        zarr_data = load_zarr_data_for_cell(zarr_url, cell_bounds, epsg_code)
        
        if zarr_data is None:
            continue
        
        # Check if clipped data is empty
        if zarr_data['b03'].size == 0 or zarr_data['b11'].size == 0:
            continue
        
        # Compute NDSI = (B03 - B11) / (B03 + B11)
        b03 = zarr_data['b03'].astype(float)
        b11 = zarr_data['b11'].astype(float)
        
        # Resample B11 to B03 resolution (20m -> 10m)
        b11_resampled = b11.interp(x=b03.x, y=b03.y, method='nearest')
        
        ndsi = (b03 - b11_resampled) / (b03 + b11_resampled + 1e-8)
        
        # Reproject NDSI from UTM to ISN2004 to ensure consistent grid
        minx, miny, maxx, maxy = cell_bounds
        target_x = np.arange(minx, maxx, 10)
        target_y = np.arange(miny, maxy, 10)
        
        product_epsg = zarr_data['metadata'].get('properties', {}).get('proj:epsg')
        ndsi_with_crs = ndsi.rio.write_crs(f"EPSG:{product_epsg}")
        ndsi_with_crs = ndsi_with_crs.rio.write_nodata(np.nan)
        
        ndsi_isn2004 = ndsi_with_crs.rio.reproject(
            f"EPSG:{epsg_code}",
            shape=(len(target_y), len(target_x)),
            transform=from_bounds(minx, miny, maxx, maxy, len(target_x), len(target_y)),
            nodata=np.nan
        )
        
        ndsi_list.append(ndsi_isn2004)
        scenes_loaded += 1
    
    if scenes_loaded == 0:
        return None
    
    # Stack scenes and compute median
    if scenes_loaded == 1:
        ndsi_median = ndsi_list[0]
    else:
        ndsi_stacked = xr.concat(ndsi_list, dim='time')
        ndsi_median = ndsi_stacked.median(dim='time')
    
    # Compute NDSI statistics
    ndsi_flat = ndsi_median.values.flatten()
    ndsi_flat_valid = ndsi_flat[~np.isnan(ndsi_flat)]
    
    if len(ndsi_flat_valid) == 0:
        return None
    
    # Classify snow/ice pixels
    snow_mask = ndsi_median >= ndsi_threshold
    snow_pixels = int(snow_mask.sum().values)
    
    # Calculate percentage based on VALID pixels only (excluding NaN from clouds/masks)
    valid_pixels = int((~np.isnan(ndsi_median.values)).sum())
    snow_percentage = 100.0 * snow_pixels / valid_pixels if valid_pixels > 0 else 0
    
    return {
        'ndsi_median': ndsi_median,
        'snow_mask': snow_mask,
        'snow_percentage': snow_percentage,
        'scene_count': scenes_loaded,
        'ndsi_stats': {
            'min': float(np.nanmin(ndsi_flat_valid)),
            'max': float(np.nanmax(ndsi_flat_valid)),
            'mean': float(np.nanmean(ndsi_flat_valid))
        }
    }


def apply_spatial_expansion(grid, cell_id, snow_percentage, threshold):
    """
    Apply spatial expansion: Add neighboring cells as candidates if snow coverage exceeds threshold.
    
    Checks all 4-connected neighbors (north, south, east, west) and marks them as candidates
    if they are not already candidates or processed.
    
    Parameters:
    -----------
    grid : gpd.GeoDataFrame
        Grid with all cells
    cell_id : int
        ID of the cell to expand from
    snow_percentage : float
        Snow coverage percentage (0-100)
    threshold : float
        Threshold for expansion (0-1, e.g., 0.30 for 30%)
    
    Returns:
    --------
    int : Number of new candidates added
    """
    # Check if snow coverage exceeds threshold
    if snow_percentage < threshold * 100:
        return 0
    
    # Get current cell geometry
    current_cell = grid[grid['cell_id'] == cell_id].iloc[0]
    current_geom = current_cell.geometry
    bounds = current_geom.bounds
    
    # Calculate neighbor positions (4-connected: north, south, east, west)
    cell_size = bounds[2] - bounds[0]
    center_x = (bounds[0] + bounds[2]) / 2
    center_y = (bounds[1] + bounds[3]) / 2
    
    neighbor_positions = [
        (center_x, center_y + cell_size),  # North
        (center_x, center_y - cell_size),  # South
        (center_x + cell_size, center_y),  # East
        (center_x - cell_size, center_y)   # West
    ]
    
    new_candidates = 0
    
    # Check each neighbor
    for nx, ny in neighbor_positions:
        # Find neighbor cell containing this point
        neighbor = grid[grid.geometry.contains(Point(nx, ny))]
        
        if len(neighbor) == 0:
            continue
        
        neighbor_id = neighbor.iloc[0]['cell_id']
        
        # Add as candidate if not already candidate or processed
        if not neighbor.iloc[0]['is_candidate'] and not neighbor.iloc[0].get('is_processed', False):
            grid_idx = grid[grid['cell_id'] == neighbor_id].index[0]
            grid.at[grid_idx, 'is_candidate'] = True
            new_candidates += 1
    
    return new_candidates


# =============================================================================
# MAIN MONITORING FUNCTION
# =============================================================================

def run_glacier_monitoring(seeds, config, verbose=True, output_dir=None):
    """
    Execute complete glacier monitoring algorithm with spatial expansion.
    
    This function implements the full algorithm:
    1. Create grid and mark initial candidates
    2. Iterate: Process candidates, compute NDSI, apply spatial expansion
    3. Combine results into unified snow/ice mask
    
    Parameters:
    -----------
    seeds : gpd.GeoDataFrame
        Glacier seed points in ISN2004 projection
    config : dict
        Configuration dictionary with all parameters
    verbose : bool
        If True, print progress information
    output_dir : Path, optional
        Output directory for low-memory mode (writes NDSI tiles to disk)
    
    Returns:
    --------
    dict : Results containing:
        - 'ndsi_combined': xr.DataArray with combined NDSI grid (None in low-memory mode)
        - 'snow_mask_combined': xr.DataArray with final snow/ice mask (None in low-memory mode)
        - 'grid': gpd.GeoDataFrame with processed cells
        - 'statistics': dict with processing statistics
        - 'tile_dir': Path to saved tiles (only in low-memory mode)
    """
    low_memory = config.get('low_memory', False)
    
    if verbose:
        print("=" * 80)
        print("GLACIER MONITORING ALGORITHM - SPATIAL EXPANSION")
        if low_memory:
            print("(LOW MEMORY MODE: Writing tiles to disk)")
        print("=" * 80)
    
    # Create temp directory for tiles in low-memory mode
    tile_dir = None
    if low_memory and output_dir:
        tile_dir = Path(output_dir) / "ndsi_tiles"
        tile_dir.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"   Tile directory: {tile_dir}")
    
    # Step 1: Create grid and mark initial candidates
    if verbose:
        print("\n[Step 1] Creating grid and marking initial candidates...")
    
    grid = create_grid(config['bounding_box'], config['grid_size'], config['epsg_iceland'])
    grid = mark_candidate_cells(grid, seeds)
    grid['is_processed'] = False
    grid['snow_percentage'] = None
    
    # Only store NDSI in grid if NOT in low-memory mode
    if not low_memory:
        grid['ndsi_median'] = None
    
    initial_candidates = grid['is_candidate'].sum()
    
    if verbose:
        print(f"   Grid created: {len(grid)} cells")
        print(f"   Initial candidates (cells with seeds): {initial_candidates}")
        if low_memory:
            print(f"   Memory mode: LOW (tiles saved to disk)")
        else:
            print(f"   Memory mode: NORMAL (results in RAM)")
    
    # Prepare result storage (only used in normal mode)
    ndsi_results = {} if not low_memory else None
    cell_bounds_dict = {}
    
    # Step 2: Iteration loop with spatial expansion
    iteration = 0
    total_processed = 0
    total_expansion_adds = 0
    
    while True:
        iteration += 1
        
        # Get unprocessed candidates
        unprocessed = grid[(grid['is_candidate']) & (~grid['is_processed'])]
        
        if len(unprocessed) == 0:
            if verbose:
                print(f"\n[Iteration {iteration}] No more candidates to process. Algorithm complete!")
            break
        
        if iteration > config['max_iterations']:
            if verbose:
                print(f"\n[Iteration {iteration}] Reached maximum iterations. Stopping.")
            break
        
        if verbose:
            print(f"\n[Iteration {iteration}] Processing {len(unprocessed)} candidate cells...")
        
        iteration_adds = 0
        cells_with_snow = 0
        
        for idx, cell_row in unprocessed.iterrows():
            cell_id = cell_row['cell_id']
            cell_geom = cell_row.geometry
            cell_bounds = cell_geom.bounds
            
            if verbose:
                print(f"  Cell {cell_id}...", end=" ")
            
            # Query STAC for this cell
            items = query_stac_for_cell(
                cell=cell_row,
                date_start=config['date_start'],
                date_end=config['date_end'],
                epsg_code=config['epsg_iceland'],
                stac_url=config['stac_url'],
                verbose=False
            )
            
            if len(items) == 0:
                if verbose:
                    print("no scenes found")
                grid_idx = grid[grid['cell_id'] == cell_id].index[0]
                grid.at[grid_idx, 'is_processed'] = True
                continue
            
            # Compute NDSI for this cell
            result = compute_median_ndsi_for_cell(
                stac_items=items,
                cell_bounds=cell_bounds,
                epsg_code=config['epsg_iceland'],
                ndsi_threshold=config['ndsi_threshold'],
                max_scenes=config['max_scenes']
            )
            
            if result is None:
                if verbose:
                    print("NDSI computation failed")
                grid_idx = grid[grid['cell_id'] == cell_id].index[0]
                grid.at[grid_idx, 'is_processed'] = True
                gc.collect()  # Free memory
                continue
            
            # Store result - either in memory or on disk
            snow_pct = result['snow_percentage']
            cell_bounds_dict[cell_id] = cell_bounds
            
            if low_memory and tile_dir:
                # Save NDSI tile to disk immediately, don't keep in memory
                tile_path = tile_dir / f"ndsi_cell_{cell_id}.tif"
                ndsi_da = result['ndsi_median']
                ndsi_da.rio.write_crs(f"EPSG:{config['epsg_iceland']}", inplace=True)
                ndsi_da.rio.to_raster(tile_path, driver='GTiff', compress='lzw')
                del ndsi_da, result  # Free memory immediately
            else:
                # Store in memory (original behavior)
                ndsi_results[cell_id] = result['ndsi_median']
                grid_idx = grid[grid['cell_id'] == cell_id].index[0]
                grid.at[grid_idx, 'ndsi_median'] = result['ndsi_median']
            
            # Add snow percentage to grid
            grid_idx = grid[grid['cell_id'] == cell_id].index[0]
            grid.at[grid_idx, 'snow_percentage'] = snow_pct
            
            if verbose:
                print(f"snow coverage: {snow_pct:.1f}%")
            
            # Apply spatial expansion
            if snow_pct > 0:
                cells_with_snow += 1
                new_adds = apply_spatial_expansion(
                    grid, cell_id, snow_pct, threshold=config['snow_percentage_threshold']
                )
                if new_adds > 0 and verbose:
                    print(f"    Added {new_adds} neighbors")
                iteration_adds += new_adds
            
            # Mark as processed
            grid.at[grid_idx, 'is_processed'] = True
            total_processed += 1
            
            # Force garbage collection to free memory
            gc.collect()
        
        total_expansion_adds += iteration_adds
        
        if verbose:
            print(f"   Summary: {len(unprocessed)} processed, {cells_with_snow} with snow, {iteration_adds} new candidates")
    
    # Step 3: Combine results or calculate statistics from tiles
    if low_memory:
        # In low-memory mode, we don't combine - just calculate stats from grid
        if verbose:
            print(f"\n[Step 3] Calculating statistics (low-memory mode - tiles saved separately)...")
        
        # Calculate statistics from the grid's snow_percentage column
        processed_grid = grid[grid['is_processed'] == True]
        total_snow_pct = processed_grid['snow_percentage'].mean() if len(processed_grid) > 0 else 0
        
        # Estimate pixels based on grid size (10km = 1000 pixels at 10m resolution)
        pixels_per_cell = (config['grid_size'] / 10) ** 2
        valid_pixels = int(total_processed * pixels_per_cell)
        snow_pixels = int(valid_pixels * (total_snow_pct / 100)) if total_snow_pct else 0
        
        statistics = {
            'total_cells_processed': total_processed,
            'initial_candidates': int(initial_candidates),
            'expansion_added_cells': total_expansion_adds,
            'iterations': iteration,
            'valid_pixels': valid_pixels,
            'snow_ice_pixels': snow_pixels,
            'snow_ice_coverage_km2': (snow_pixels * 10 * 10) / 1e6,
            'total_valid_area_km2': (valid_pixels * 10 * 10) / 1e6,
            'snow_ice_percentage': total_snow_pct,
            'mode': 'low_memory',
            'tile_directory': str(tile_dir) if tile_dir else None
        }
        
        ndsi_combined = None
        snow_mask_combined = None
        
    else:
        # Normal mode: combine all results in memory
        if verbose:
            print(f"\n[Step 3] Combining {len(ndsi_results)} NDSI results into unified grid...")
        
        if len(ndsi_results) == 0:
            print("ERROR: No valid NDSI results to combine!")
            return None
        
        # Determine combined grid extent
        all_bounds = list(cell_bounds_dict.values())
        global_minx = min(b[0] for b in all_bounds)
        global_miny = min(b[1] for b in all_bounds)
        global_maxx = max(b[2] for b in all_bounds)
        global_maxy = max(b[3] for b in all_bounds)
        
        # Create combined grid (10m resolution)
        combined_x = np.arange(global_minx, global_maxx, 10)
        combined_y = np.arange(global_miny, global_maxy, 10)
        combined_ndsi = np.full((len(combined_y), len(combined_x)), np.nan, dtype=np.float32)  # Use float32 to save memory
        
        # Fill in NDSI values from each cell
        for cell_id, ndsi_array in ndsi_results.items():
            bounds = cell_bounds_dict[cell_id]
            minx, miny, maxx, maxy = bounds
            
            # Find indices in combined grid
            x_start = int((minx - global_minx) / 10)
            x_end = int((maxx - global_minx) / 10)
            y_start = int((miny - global_miny) / 10)
            y_end = int((maxy - global_miny) / 10)
            
            # Insert NDSI values
            ndsi_values = ndsi_array.values
            combined_ndsi[y_start:y_end, x_start:x_end] = ndsi_values
        
        # Free the individual results
        del ndsi_results
        gc.collect()
        
        # Create xarray DataArray
        ndsi_combined = xr.DataArray(
            combined_ndsi,
            dims=['y', 'x'],
            coords={'x': combined_x, 'y': combined_y},
            attrs={
                'crs': f'EPSG:{config["epsg_iceland"]}',
                'description': 'Combined median NDSI from all processed cells',
                'algorithm': 'Zarr-optimized glacier monitoring with spatial expansion'
            }
        )
        
        # Create final snow/ice mask
        snow_mask_combined = ndsi_combined >= config['ndsi_threshold']
        
        # Calculate statistics
        valid_pixels = int(np.sum(~np.isnan(combined_ndsi)))
        snow_pixels = int(np.sum(snow_mask_combined.values))
        snow_area_km2 = (snow_pixels * 10 * 10) / 1e6
        total_area_km2 = (valid_pixels * 10 * 10) / 1e6
        
        statistics = {
            'total_cells_processed': total_processed,
            'initial_candidates': int(initial_candidates),
            'expansion_added_cells': total_expansion_adds,
            'iterations': iteration,
            'valid_pixels': valid_pixels,
            'snow_ice_pixels': snow_pixels,
            'snow_ice_coverage_km2': snow_area_km2,
            'total_valid_area_km2': total_area_km2,
            'snow_ice_percentage': 100.0 * snow_pixels / valid_pixels if valid_pixels > 0 else 0,
            'mode': 'normal'
        }
    
    # Print final statistics
    if verbose:
        print("\n" + "=" * 80)
        print("ALGORITHM COMPLETED - FINAL STATISTICS")
        print("=" * 80)
        print(f"Mode:                         {statistics.get('mode', 'normal')}")
        print(f"Total cells processed:        {statistics['total_cells_processed']}")
        print(f"  - Initial candidates:       {statistics['initial_candidates']}")
        print(f"  - Added by expansion:       {statistics['expansion_added_cells']}")
        print(f"Total iterations:             {statistics['iterations']}")
        print(f"Valid pixels analyzed:        {statistics['valid_pixels']:,}")
        print(f"Snow/Ice pixels detected:     {statistics['snow_ice_pixels']:,}")
        print(f"Snow/Ice area:                {statistics['snow_ice_coverage_km2']:.2f} km²")
        print(f"Total analyzed area:          {statistics['total_valid_area_km2']:.2f} km²")
        print(f"Snow/Ice coverage:            {statistics['snow_ice_percentage']:.2f}%")
        if low_memory and tile_dir:
            print(f"Tiles saved to:               {tile_dir}")
        print("=" * 80)
    
    return {
        'ndsi_combined': ndsi_combined,
        'snow_mask_combined': snow_mask_combined,
        'grid': grid,
        'statistics': statistics,
        'tile_dir': tile_dir
    }


# =============================================================================
# OUTPUT EXPORT FUNCTIONS
# =============================================================================

def export_results(results, output_dir, config):
    """
    Export all results to output directory.
    
    Exports:
    - Grid with processed cells as GeoJSON
    - Combined NDSI as GeoTIFF (only in normal mode)
    - Snow/Ice mask as GeoTIFF (only in normal mode)
    - Statistics as JSON
    
    Parameters:
    -----------
    results : dict
        Results from run_glacier_monitoring
    output_dir : Path
        Output directory path
    config : dict
        Configuration dictionary
    """
    low_memory = config.get('low_memory', False)
    
    print("\n" + "=" * 80)
    print("EXPORTING RESULTS")
    print("=" * 80)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Export grid with processed cells
    grid_output = output_dir / f"processed_grid_{timestamp}.geojson"
    grid_export = results['grid'][results['grid']['is_processed']].copy()
    
    # Remove ndsi_median column if it exists (contains xarray objects, not JSON serializable)
    if 'ndsi_median' in grid_export.columns:
        grid_export = grid_export.drop(columns=['ndsi_median'])
    
    grid_export.to_file(grid_output, driver='GeoJSON')
    print(f"[1/4] Grid exported: {grid_output}")
    
    if low_memory:
        # In low-memory mode, tiles are already saved
        tile_dir = results.get('tile_dir')
        print(f"[2/4] NDSI tiles already saved: {tile_dir}")
        print(f"[3/4] Snow masks: Generate from tiles using NDSI threshold {config['ndsi_threshold']}")
    else:
        # 2. Export combined NDSI as GeoTIFF
        if results['ndsi_combined'] is not None:
            ndsi_output = output_dir / f"ndsi_combined_{timestamp}.tif"
            results['ndsi_combined'].rio.write_crs(f"EPSG:{config['epsg_iceland']}", inplace=True)
            results['ndsi_combined'].rio.to_raster(ndsi_output, driver='GTiff', compress='lzw')
            print(f"[2/4] NDSI raster exported: {ndsi_output}")
        else:
            print(f"[2/4] NDSI raster: skipped (not available)")
        
        # 3. Export snow mask as GeoTIFF
        if results['snow_mask_combined'] is not None:
            mask_output = output_dir / f"snow_mask_{timestamp}.tif"
            results['snow_mask_combined'].astype(np.uint8).rio.to_raster(mask_output, driver='GTiff', compress='lzw')
            print(f"[3/4] Snow mask exported: {mask_output}")
        else:
            print(f"[3/4] Snow mask: skipped (not available)")
    
    # 4. Export statistics as JSON
    stats_output = output_dir / f"statistics_{timestamp}.json"
    stats_with_config = {
        'configuration': {k: v for k, v in config.items() if not callable(v)},
        'results': results['statistics']
    }
    with open(stats_output, 'w') as f:
        json.dump(stats_with_config, f, indent=2)
    print(f"[4/4] Statistics exported: {stats_output}")
    
    print("=" * 80)
    print("All results exported successfully!")
    print("=" * 80)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """
    Main entry point for the glacier monitoring script.
    
    Parses command-line arguments and executes the monitoring algorithm.
    """
    parser = argparse.ArgumentParser(
        description='Glacier Monitoring using Sentinel-2 L2A Data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default parameters
  python glacier_monitoring.py --seeds data/Iceland_Seeds.geojson --output results
  
  # Process single seed by FID
  python glacier_monitoring.py --seeds data/Iceland_Seeds.geojson --fid 5 --output results
  
  # Custom time period
  python glacier_monitoring.py --seeds data/Iceland_Seeds.geojson --output results \\
    --date-start 2025-07-01 --date-end 2025-08-31
  
  # Custom thresholds
  python glacier_monitoring.py --seeds data/Iceland_Seeds.geojson --output results \\
    --ndsi-threshold 0.40 --snow-threshold 0.25
        """
    )
    
    # Required arguments
    parser.add_argument('--seeds', required=True, type=str,
                        help='Path to glacier seeds GeoJSON file')
    parser.add_argument('--output', required=True, type=str,
                        help='Output directory for results')
    
    # Optional arguments
    parser.add_argument('--fid', type=int, default=None,
                        help='FID of specific seed to process (default: process all seeds)')
    parser.add_argument('--date-start', type=str, default=DEFAULT_CONFIG['date_start'],
                        help=f'Start date (YYYY-MM-DD, default: {DEFAULT_CONFIG["date_start"]})')
    parser.add_argument('--date-end', type=str, default=DEFAULT_CONFIG['date_end'],
                        help=f'End date (YYYY-MM-DD, default: {DEFAULT_CONFIG["date_end"]})')
    parser.add_argument('--grid-size', type=int, default=DEFAULT_CONFIG['grid_size'],
                        help=f'Grid cell size in meters (default: {DEFAULT_CONFIG["grid_size"]})')
    parser.add_argument('--ndsi-threshold', type=float, default=DEFAULT_CONFIG['ndsi_threshold'],
                        help=f'NDSI threshold for snow/ice (default: {DEFAULT_CONFIG["ndsi_threshold"]})')
    parser.add_argument('--snow-threshold', type=float, default=DEFAULT_CONFIG['snow_percentage_threshold'],
                        help=f'Snow coverage threshold for expansion (default: {DEFAULT_CONFIG["snow_percentage_threshold"]})')
    parser.add_argument('--max-scenes', type=int, default=DEFAULT_CONFIG['max_scenes'],
                        help=f'Maximum scenes per cell (default: {DEFAULT_CONFIG["max_scenes"]})')
    parser.add_argument('--max-iterations', type=int, default=DEFAULT_CONFIG['max_iterations'],
                        help=f'Maximum iterations (default: {DEFAULT_CONFIG["max_iterations"]})')
    parser.add_argument('--low-memory', action='store_true', default=DEFAULT_CONFIG['low_memory'],
                        help='Enable low-memory mode: saves tiles to disk instead of RAM (recommended for large areas)')
    parser.add_argument('--no-low-memory', action='store_true',
                        help='Disable low-memory mode: keep all results in RAM')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')
    
    args = parser.parse_args()
    
    # Load configuration
    config = DEFAULT_CONFIG.copy()
    config['date_start'] = args.date_start
    config['date_end'] = args.date_end
    config['grid_size'] = args.grid_size
    config['ndsi_threshold'] = args.ndsi_threshold
    config['snow_percentage_threshold'] = args.snow_threshold
    config['max_scenes'] = args.max_scenes
    config['max_iterations'] = args.max_iterations
    
    # Handle low-memory mode flags
    if args.no_low_memory:
        config['low_memory'] = False
    else:
        config['low_memory'] = args.low_memory
    
    verbose = not args.quiet
    
    # Print header
    if verbose:
        print("\n" + "=" * 80)
        print("GLACIER MONITORING SCRIPT")
        print("=" * 80)
        print(f"Seeds file:        {args.seeds}")
        print(f"Output directory:  {args.output}")
        print(f"Time period:       {args.date_start} to {args.date_end}")
        print(f"Grid size:         {args.grid_size} m")
        print(f"NDSI threshold:    {args.ndsi_threshold}")
        print(f"Snow threshold:    {args.snow_threshold * 100}%")
        print(f"Max scenes/cell:   {args.max_scenes}")
        print(f"Low-memory mode:   {'ON' if config['low_memory'] else 'OFF'}")
        print("=" * 80)
    
    # Load seeds
    if verbose:
        print("\nLoading glacier seeds...")
    
    try:
        seeds = gpd.read_file(args.seeds)
        
        # Filter by FID if specified
        if args.fid is not None:
            if 'fid' not in seeds.columns:
                print(f"ERROR: Seeds file does not contain 'fid' column")
                return 1
            
            seeds = seeds[seeds['fid'] == args.fid]
            
            if len(seeds) == 0:
                print(f"ERROR: No seed found with fid={args.fid}")
                return 1
            
            if verbose:
                print(f"  Filtered to single seed: fid={args.fid}")
        
        # Transform to ISN2004 if necessary
        if seeds.crs.to_epsg() != config['epsg_iceland']:
            seeds = seeds.to_crs(epsg=config['epsg_iceland'])
            if verbose:
                print(f"  Transformed to EPSG:{config['epsg_iceland']}")
        
        if verbose:
            print(f"  Loaded {len(seeds)} seed(s)")
    
    except Exception as e:
        print(f"ERROR: Failed to load seeds from {args.seeds}: {e}")
        return 1
    
    # Prepare output directory
    output_dir = Path(args.output)
    
    # Run monitoring algorithm
    try:
        results = run_glacier_monitoring(
            seeds, 
            config, 
            verbose=verbose, 
            output_dir=output_dir  # Pass output dir for low-memory tile saving
        )
        
        if results is None:
            print("ERROR: Monitoring algorithm failed")
            return 1
    
    except Exception as e:
        print(f"ERROR: Monitoring algorithm failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Export results
    try:
        export_results(results, output_dir, config)
    
    except Exception as e:
        print(f"ERROR: Failed to export results: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    if verbose:
        print("\nGlacier monitoring completed successfully!")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
