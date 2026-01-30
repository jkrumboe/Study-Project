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
import time
import argparse
import warnings
import json
from datetime import datetime
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# Progress bar
from tqdm import tqdm

# Geospatial libraries
import geopandas as gpd
from shapely.geometry import box, Point
from pyproj import Transformer

# Data processing
import xarray as xr
import numpy as np
import pandas as pd
import rioxarray
import rasterio
from rasterio.transform import from_bounds
from rasterio.enums import Resampling

# STAC API for Sentinel-2 data
from pystac_client import Client

# Retry logic
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import requests
import threading
import urllib3

# Suppress SSL warnings for Proxmox self-signed certs
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Global caches to avoid redundant operations
_failed_zarr_urls = set()  # Track URLs that failed to load
_scene_cache = {}  # Cache STAC results by bbox tuple
_stac_catalog = None  # Reuse STAC connection


# =============================================================================
# CELL TIMING TRACKER
# =============================================================================

class CellTimingTracker:
    """
    Track processing time for each cell and each processing phase.
    
    Provides detailed timing breakdown for:
    - STAC query time
    - Zarr data loading time
    - NDSI computation time
    - Tile saving time (in low-memory mode)
    - Spatial expansion time
    """
    
    def __init__(self):
        """Initialize the timing tracker."""
        self.cell_timings = []  # List of timing records per cell
        self.current_cell = None  # Current cell timing in progress
        self.iteration_timings = []  # Timing summaries per iteration
        
    def start_cell(self, cell_id):
        """Start timing for a new cell."""
        self.current_cell = {
            'cell_id': cell_id,
            'start_time': time.time(),
            'stac_query_time': 0,
            'zarr_load_time': 0,
            'ndsi_compute_time': 0,
            'tile_save_time': 0,
            'expansion_time': 0,
            'total_time': 0,
            'scenes_found': 0,
            'scenes_loaded': 0,
            'snow_percentage': None
        }
        
    def record_stac_query(self, duration, scenes_found=0):
        """Record STAC query time."""
        if self.current_cell:
            self.current_cell['stac_query_time'] = duration
            self.current_cell['scenes_found'] = scenes_found
            
    def record_zarr_load(self, duration, scenes_loaded=0):
        """Record Zarr loading time."""
        if self.current_cell:
            self.current_cell['zarr_load_time'] = duration
            self.current_cell['scenes_loaded'] = scenes_loaded
            
    def record_ndsi_compute(self, duration):
        """Record NDSI computation time."""
        if self.current_cell:
            self.current_cell['ndsi_compute_time'] = duration
            
    def record_tile_save(self, duration):
        """Record tile saving time."""
        if self.current_cell:
            self.current_cell['tile_save_time'] = duration
            
    def record_expansion(self, duration):
        """Record spatial expansion time."""
        if self.current_cell:
            self.current_cell['expansion_time'] = duration
            
    def end_cell(self, snow_percentage=None):
        """End timing for current cell and store the record."""
        if self.current_cell:
            self.current_cell['total_time'] = time.time() - self.current_cell['start_time']
            self.current_cell['snow_percentage'] = snow_percentage
            self.cell_timings.append(self.current_cell)
            self.current_cell = None
            
    def start_iteration(self, iteration_num, num_cells):
        """Start timing for an iteration."""
        self._current_iteration = {
            'iteration': iteration_num,
            'num_cells': num_cells,
            'start_time': time.time()
        }
        
    def end_iteration(self, cells_processed, cells_with_snow, new_candidates):
        """End timing for an iteration."""
        if hasattr(self, '_current_iteration'):
            self._current_iteration['total_time'] = time.time() - self._current_iteration['start_time']
            self._current_iteration['cells_processed'] = cells_processed
            self._current_iteration['cells_with_snow'] = cells_with_snow
            self._current_iteration['new_candidates'] = new_candidates
            self.iteration_timings.append(self._current_iteration)
            
    def get_summary(self):
        """Get summary statistics of all cell timings."""
        if not self.cell_timings:
            return {}
            
        total_cells = len(self.cell_timings)
        
        # Calculate aggregated times
        total_stac = sum(c['stac_query_time'] for c in self.cell_timings)
        total_zarr = sum(c['zarr_load_time'] for c in self.cell_timings)
        total_ndsi = sum(c['ndsi_compute_time'] for c in self.cell_timings)
        total_tile = sum(c['tile_save_time'] for c in self.cell_timings)
        total_expansion = sum(c['expansion_time'] for c in self.cell_timings)
        total_cell_time = sum(c['total_time'] for c in self.cell_timings)
        
        # Calculate per-cell averages
        avg_stac = total_stac / total_cells if total_cells > 0 else 0
        avg_zarr = total_zarr / total_cells if total_cells > 0 else 0
        avg_ndsi = total_ndsi / total_cells if total_cells > 0 else 0
        avg_tile = total_tile / total_cells if total_cells > 0 else 0
        avg_expansion = total_expansion / total_cells if total_cells > 0 else 0
        avg_total = total_cell_time / total_cells if total_cells > 0 else 0
        
        # Calculate time percentages
        total_processing = total_stac + total_zarr + total_ndsi + total_tile + total_expansion
        pct_stac = 100 * total_stac / total_processing if total_processing > 0 else 0
        pct_zarr = 100 * total_zarr / total_processing if total_processing > 0 else 0
        pct_ndsi = 100 * total_ndsi / total_processing if total_processing > 0 else 0
        pct_tile = 100 * total_tile / total_processing if total_processing > 0 else 0
        pct_expansion = 100 * total_expansion / total_processing if total_processing > 0 else 0
        
        # Find slowest cells
        sorted_by_time = sorted(self.cell_timings, key=lambda x: x['total_time'], reverse=True)
        slowest_cells = sorted_by_time[:5] if len(sorted_by_time) >= 5 else sorted_by_time
        
        return {
            'total_cells_timed': total_cells,
            'total_times': {
                'stac_query_seconds': round(total_stac, 2),
                'zarr_load_seconds': round(total_zarr, 2),
                'ndsi_compute_seconds': round(total_ndsi, 2),
                'tile_save_seconds': round(total_tile, 2),
                'spatial_expansion_seconds': round(total_expansion, 2),
                'total_cell_processing_seconds': round(total_cell_time, 2)
            },
            'average_times_per_cell': {
                'stac_query_seconds': round(avg_stac, 3),
                'zarr_load_seconds': round(avg_zarr, 3),
                'ndsi_compute_seconds': round(avg_ndsi, 3),
                'tile_save_seconds': round(avg_tile, 3),
                'spatial_expansion_seconds': round(avg_expansion, 3),
                'total_seconds': round(avg_total, 3)
            },
            'time_distribution_percent': {
                'stac_query': round(pct_stac, 1),
                'zarr_load': round(pct_zarr, 1),
                'ndsi_compute': round(pct_ndsi, 1),
                'tile_save': round(pct_tile, 1),
                'spatial_expansion': round(pct_expansion, 1)
            },
            'slowest_cells': [
                {
                    'cell_id': c['cell_id'],
                    'total_time': round(c['total_time'], 2),
                    'stac_query': round(c['stac_query_time'], 2),
                    'zarr_load': round(c['zarr_load_time'], 2),
                    'ndsi_compute': round(c['ndsi_compute_time'], 2),
                    'scenes_loaded': c['scenes_loaded']
                }
                for c in slowest_cells
            ],
            'iteration_timings': [
                {
                    'iteration': it['iteration'],
                    'cells_processed': it.get('cells_processed', 0),
                    'total_time_seconds': round(it.get('total_time', 0), 2),
                    'avg_time_per_cell': round(it.get('total_time', 0) / it.get('cells_processed', 1), 2) if it.get('cells_processed', 0) > 0 else 0
                }
                for it in self.iteration_timings
            ],
            'per_cell_timings': self.cell_timings
        }


# Global timing tracker instance
_cell_timing_tracker = None


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    'epsg_iceland': 5325,  # ISN2004 / Lambert 2004 (Iceland's national projection)
    'bounding_box': [1400000, 100000, 2000000, 500000],  # [minx, miny, maxx, maxy] in ISN2004
    'grid_size': 10000,  # Grid cell size in meters (1 km)
    'ndsi_threshold': 0.42,  # Snow/ice classification threshold
    'snow_percentage_threshold': 0.30,  # Spatial expansion threshold (30% coverage)
    'stac_url': "https://stac.core.eopf.eodc.eu",  # EOPF STAC Catalog endpoint
    'date_start': "2025-07-01",  # Start date (YYYY-MM-DD)
    'date_end': "2025-07-31",  # End date (YYYY-MM-DD)
    'max_iterations': 2,  # Maximum iterations for spatial expansion
    'max_scenes': 100,  # Maximum scenes to process per cell
    'low_memory': True,  # Memory-efficient mode: write results to disk incrementally
    'max_cloud_cover': 50,  # Filter out scenes with more than X% cloud cover (currently disabled)
    'max_retries': 3,  # Number of retries for failed network requests
    'validate_zarr': True  # Validate Zarr store accessibility before processing
}

# Proxmox monitoring configuration
PROXMOX_CONFIG = {
    'enabled': False,  # Set to True to enable Proxmox monitoring
    'host': '192.168.2.119',
    'port': 8006,
    'node': 'think1',
    'vmid': 103,
    'api_token_id': 'monitor@think1@pam!monitorthink1',
    'api_token_secret': '109ee029-4668-45be-9572-9a1116a9ed95',
    'poll_interval': 2,  # seconds between status polls (high-resolution monitoring)
    'fetch_rrd': True,  # Also fetch RRD timeline data at end
    'rrd_timeframe': 'hour'  # hour | day | week | month | year
}


# =============================================================================
# PROXMOX MONITORING CLASS
# =============================================================================

class ProxmoxMonitor:
    """
    Monitor VM performance using Proxmox API.
    
    Collects CPU, memory, network, and disk I/O metrics during processing.
    Runs in a background thread to avoid blocking the main algorithm.
    """
    
    def __init__(self, config=None):
        """
        Initialize the Proxmox monitor.
        
        Parameters:
        -----------
        config : dict, optional
            Proxmox configuration. Uses PROXMOX_CONFIG defaults if not provided.
        """
        self.config = config or PROXMOX_CONFIG.copy()
        self.enabled = self.config.get('enabled', False)
        self.base_url = f"https://{self.config['host']}:{self.config['port']}/api2/json"
        self.headers = {
            'Authorization': f"PVEAPIToken={self.config['api_token_id']}={self.config['api_token_secret']}"
        }
        
        # Monitoring state
        self.samples = []  # High-resolution samples from status/current
        self.events = []  # Algorithm events (cell processing, iterations, etc.)
        self.start_time = None
        self.end_time = None
        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        
    def _api_get(self, endpoint):
        """Make a GET request to the Proxmox API."""
        url = f"{self.base_url}/{endpoint}"
        try:
            response = requests.get(url, headers=self.headers, verify=False, timeout=5)
            response.raise_for_status()
            return response.json().get('data', {})
        except Exception as e:
            return {'error': str(e)}
    
    def get_vm_status(self):
        """Get current VM status (CPU, memory, network, disk)."""
        node = self.config['node']
        vmid = self.config['vmid']
        return self._api_get(f"nodes/{node}/qemu/{vmid}/status/current")
    
    def get_rrd_data(self, timeframe=None):
        """Get RRD timeline data (aggregated metrics over time)."""
        node = self.config['node']
        vmid = self.config['vmid']
        tf = timeframe or self.config.get('rrd_timeframe', 'hour')
        return self._api_get(f"nodes/{node}/qemu/{vmid}/rrddata?timeframe={tf}&cf=AVERAGE")
    
    def _polling_loop(self):
        """Background thread that polls VM status at regular intervals."""
        interval = self.config.get('poll_interval', 2)
        
        while not self._stop_event.is_set():
            status = self.get_vm_status()
            if 'error' not in status:
                sample = {
                    'time': time.time(),
                    'time_iso': datetime.now().isoformat(),
                    'cpu': status.get('cpu', 0),
                    'cpu_percent': status.get('cpu', 0) * 100,
                    'mem': status.get('mem', 0),
                    'maxmem': status.get('maxmem', 0),
                    'mem_percent': 100 * status.get('mem', 0) / status.get('maxmem', 1) if status.get('maxmem') else 0,
                    'netin': status.get('netin', 0),
                    'netout': status.get('netout', 0),
                    'diskread': status.get('diskread', 0),
                    'diskwrite': status.get('diskwrite', 0),
                    'uptime': status.get('uptime', 0)
                }
                with self._lock:
                    self.samples.append(sample)
            
            self._stop_event.wait(interval)
    
    def add_event(self, event_type, details=None):
        """
        Record an algorithm event for correlation with metrics.
        
        Parameters:
        -----------
        event_type : str
            Type of event (e.g., 'iteration_start', 'cell_processed', 'stac_query')
        details : dict, optional
            Additional event details
        """
        if not self.enabled:
            return
        
        event = {
            'time': time.time(),
            'time_iso': datetime.now().isoformat(),
            'event_type': event_type,
            'details': details or {}
        }
        with self._lock:
            self.events.append(event)
    
    def start(self):
        """Start background monitoring."""
        if not self.enabled:
            return
        
        self.start_time = time.time()
        self.samples = []
        self.events = []
        self._stop_event.clear()
        
        # Record initial status
        self.add_event('monitoring_started', {'start_time': self.start_time})
        
        # Start polling thread
        self._thread = threading.Thread(target=self._polling_loop, daemon=True)
        self._thread.start()
        print(f"   Proxmox monitoring started (polling every {self.config.get('poll_interval', 2)}s)")
    
    def stop(self):
        """Stop background monitoring and collect final data."""
        if not self.enabled:
            return {}
        
        self.end_time = time.time()
        self._stop_event.set()
        
        if self._thread:
            self._thread.join(timeout=5)
        
        self.add_event('monitoring_stopped', {'end_time': self.end_time})
        
        # Fetch RRD data for the monitoring period
        rrd_data = None
        if self.config.get('fetch_rrd', True):
            rrd_data = self.get_rrd_data()
            if isinstance(rrd_data, list):
                # Filter to our time window
                rrd_data = [
                    point for point in rrd_data 
                    if self.start_time <= point.get('time', 0) <= self.end_time + 60
                ]
        
        return self.get_report(rrd_data)
    
    def get_report(self, rrd_data=None):
        """Generate a monitoring report."""
        if not self.enabled or not self.samples:
            return {}
        
        duration = (self.end_time or time.time()) - self.start_time
        
        # Calculate network transfer during monitoring
        if len(self.samples) >= 2:
            net_in_start = self.samples[0]['netin']
            net_in_end = self.samples[-1]['netin']
            net_out_start = self.samples[0]['netout']
            net_out_end = self.samples[-1]['netout']
            disk_read_start = self.samples[0]['diskread']
            disk_read_end = self.samples[-1]['diskread']
            disk_write_start = self.samples[0]['diskwrite']
            disk_write_end = self.samples[-1]['diskwrite']
        else:
            net_in_start = net_in_end = 0
            net_out_start = net_out_end = 0
            disk_read_start = disk_read_end = 0
            disk_write_start = disk_write_end = 0
        
        # Calculate statistics
        cpu_values = [s['cpu_percent'] for s in self.samples]
        mem_values = [s['mem_percent'] for s in self.samples]
        
        report = {
            'monitoring_period': {
                'start_time': self.start_time,
                'start_time_iso': datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
                'end_time': self.end_time,
                'end_time_iso': datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
                'duration_seconds': round(duration, 2),
                'duration_formatted': f"{int(duration // 60)}m {int(duration % 60)}s"
            },
            'samples_collected': len(self.samples),
            'events_recorded': len(self.events),
            'cpu_stats': {
                'min_percent': round(min(cpu_values), 2) if cpu_values else 0,
                'max_percent': round(max(cpu_values), 2) if cpu_values else 0,
                'avg_percent': round(sum(cpu_values) / len(cpu_values), 2) if cpu_values else 0
            },
            'memory_stats': {
                'min_percent': round(min(mem_values), 2) if mem_values else 0,
                'max_percent': round(max(mem_values), 2) if mem_values else 0,
                'avg_percent': round(sum(mem_values) / len(mem_values), 2) if mem_values else 0
            },
            'network_transfer': {
                'bytes_in': net_in_end - net_in_start,
                'bytes_out': net_out_end - net_out_start,
                'mb_in': round((net_in_end - net_in_start) / (1024 * 1024), 2),
                'mb_out': round((net_out_end - net_out_start) / (1024 * 1024), 2),
                'avg_speed_mbps_in': round((net_in_end - net_in_start) * 8 / duration / 1e6, 2) if duration > 0 else 0,
                'avg_speed_mbps_out': round((net_out_end - net_out_start) * 8 / duration / 1e6, 2) if duration > 0 else 0
            },
            'disk_io': {
                'bytes_read': disk_read_end - disk_read_start,
                'bytes_written': disk_write_end - disk_write_start,
                'mb_read': round((disk_read_end - disk_read_start) / (1024 * 1024), 2),
                'mb_written': round((disk_write_end - disk_write_start) / (1024 * 1024), 2)
            },
            'samples': self.samples,  # Full time-series data
            'events': self.events,  # Algorithm events for correlation
            'rrd_data': rrd_data  # Proxmox RRD data if fetched
        }
        
        return report
    
    def save_report(self, output_dir, filename_prefix='proxmox_monitoring'):
        """Save the monitoring report to JSON file."""
        if not self.enabled:
            return None
        
        report = self.get_report()
        if not report:
            return None
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = output_dir / f"{filename_prefix}_{timestamp}.json"
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return filepath


# Global monitor instance (used by run_glacier_monitoring)
_proxmox_monitor = None


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


def get_stac_catalog(stac_url):
    """
    Get or create a reusable STAC catalog connection.
    
    This avoids creating a new connection for each query, improving performance.
    """
    global _stac_catalog
    if _stac_catalog is None:
        _stac_catalog = Client.open(stac_url)
    return _stac_catalog


def validate_stac_connection(stac_url, verbose=True):
    """
    Validate that the STAC catalog is accessible before processing.
    
    Returns True if connection is successful, False otherwise.
    """
    try:
        catalog = get_stac_catalog(stac_url)
        # Try to get collections to verify connection
        collections = list(catalog.get_collections())
        if verbose:
            print(f"   STAC connection OK: {len(collections)} collections available")
        return True
    except Exception as e:
        if verbose:
            print(f"   ERROR: Cannot connect to STAC catalog: {e}")
        return False


def query_stac_for_cell(cell, date_start, date_end, epsg_code, stac_url, max_cloud_cover=50, verbose=False):
    """
    Query EOPF STAC Catalog for Sentinel-2 L2A scenes covering a grid cell.
    
    This function searches for all Sentinel-2 L2A scenes that intersect with
    the given cell's geometry during the specified time period. It handles
    coordinate transformation from ISN2004 to WGS84 (required by STAC API).
    
    Features:
    - Reuses STAC catalog connection for performance
    - Caches results by bounding box to avoid redundant queries
    - Filters by cloud cover to reduce wasted processing
    
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
    max_cloud_cover : int
        Maximum cloud cover percentage (0-100)
    verbose : bool
        If True, print progress information
    
    Returns:
    --------
    list : List of STAC items (pystac.Item objects) with Zarr data URLs
    """
    global _scene_cache
    
    # Transform cell bounds from ISN2004 to WGS84 (required by STAC API)
    cell_gdf = gpd.GeoDataFrame([cell], crs=f"EPSG:{epsg_code}")
    cell_wgs84 = cell_gdf.to_crs(epsg=4326)
    bbox_wgs84 = tuple(cell_wgs84.total_bounds)
    
    # Create cache key from bbox and date range
    cache_key = (bbox_wgs84, date_start, date_end, max_cloud_cover)
    
    # Return cached results if available
    if cache_key in _scene_cache:
        if verbose:
            print(f"  Using cached STAC results for cell {cell.get('cell_id', 'unknown')}")
        return _scene_cache[cache_key]
    
    if verbose:
        print(f"  Querying STAC for cell {cell.get('cell_id', 'unknown')}: {date_start} to {date_end}")
    
    # Reuse STAC catalog connection
    catalog = get_stac_catalog(stac_url)
    
    # Search for Sentinel-2 L2A scenes that intersect the cell
    # Cloud cover filter commented out to match old script behavior
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox_wgs84,
        datetime=[date_start, date_end]
        # query={"eo:cloud_cover": {"lte": max_cloud_cover}}  # Filter cloudy scenes - DISABLED
    )
    
    # Collect results
    items = list(search.items())
    
    # Sort by cloud cover (least cloudy first)
    items.sort(key=lambda x: x.properties.get('eo:cloud_cover', 100))
    
    if verbose:
        print(f"    Found {len(items)} scenes")
    
    # Cache results
    _scene_cache[cache_key] = items
    
    return items


def load_zarr_data_for_cell(zarr_url, cell_bounds_isn2004, cell_epsg_isn2004, max_retries=3, verbose_errors=True):
    """
    Load and extract Zarr data for a specific cell using ESA best practices.
    
    This function handles coordinate transformation from ISN2004 to the product's
    UTM zone, loads only relevant data chunks, and applies quality masking.
    
    Features:
    - Skips URLs that previously failed (to avoid repeated errors)
    - Retry logic for transient network failures
    - Suppresses duplicate error messages
    
    Parameters:
    -----------
    zarr_url : str
        URL to the EOPF Sentinel-2 L2A Zarr store
    cell_bounds_isn2004 : tuple
        Cell bounds in ISN2004 coordinates (minx, miny, maxx, maxy)
    cell_epsg_isn2004 : int
        EPSG code of input coordinates
    max_retries : int
        Number of retry attempts for transient failures
    verbose_errors : bool
        If False, suppress error messages for known failed URLs
    
    Returns:
    --------
    dict or None : Dictionary containing:
        - 'b03': Green band (10m resolution) as xr.DataArray
        - 'b11': SWIR1 band (20m resolution) as xr.DataArray
        - 'valid_mask': Quality mask (20m resolution)
        - 'metadata': STAC discovery metadata
    """
    global _failed_zarr_urls
    
    # Skip URLs that have previously failed
    if zarr_url in _failed_zarr_urls:
        return None
    
    # Retry logic for transient failures
    last_error = None
    for attempt in range(max_retries):
        try:
            # Open Zarr store as DataTree (ESA recommended approach)
            dt = xr.open_datatree(zarr_url, engine='zarr', chunks={})
            break  # Success, exit retry loop
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))  # Exponential backoff
            continue
    else:
        # All retries failed
        _failed_zarr_urls.add(zarr_url)
        if verbose_errors:
            scene_name = zarr_url.split('/')[-1]
            print(f"    ✗ {scene_name}: {last_error}")
        return None
    
    try:
        
        # Extract metadata
        metadata = dt.attrs.get('stac_discovery', {})
        product_epsg = metadata.get('properties', {}).get('proj:epsg')
        
        # If EPSG is not in metadata, infer it from the tile ID in the URL
        # URL format includes tile ID like T28WDS (UTM zone 28, latitude band W, grid square DS)
        if product_epsg is None:
            # Extract tile ID from URL (e.g., "...T28WDS_20251019..." -> "T28WDS")
            import re
            tile_match = re.search(r'_T(\d{2})([A-Z])([A-Z]{2})_', zarr_url)
            if tile_match:
                zone = int(tile_match.group(1))
                lat_band = tile_match.group(2)
                # Northern hemisphere: C-X (excluding I and O), Southern: A-M (excluding I and O)
                # Iceland is in bands V-W, which is Northern hemisphere
                if lat_band >= 'N':
                    product_epsg = 32600 + zone  # UTM Northern hemisphere
                else:
                    product_epsg = 32700 + zone  # UTM Southern hemisphere
        
        # Transform coordinates from ISN2004 to product UTM zone
        if product_epsg is not None and product_epsg != cell_epsg_isn2004:
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
        
        # Apply quality mask to B11 only (matching old script behavior)
        # Note: B03 is NOT masked to match glacier_monitoring_old.py
        b11_clipped = b11_clipped.where(valid_clipped)
        
        # Check if data is empty
        if b03_clipped.size == 0 or b11_clipped.size == 0:
            return None
        
        return {
            'b03': b03_clipped,
            'b11': b11_clipped,
            'valid_mask': valid_clipped,
            'metadata': metadata,
            'product_epsg': product_epsg  # Include the resolved EPSG code
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
        - 'zarr_load_time': Total time spent loading Zarr data
    """
    ndsi_list = []
    scenes_loaded = 0
    total_zarr_load_time = 0  # Track Zarr loading time
    
    # Load and process each scene
    for idx, item in enumerate(stac_items[:max_scenes]):
        zarr_url = item.assets["product"].href
        
        # Load Zarr data (handles coordinate transformation automatically) - timed
        zarr_load_start = time.time()
        zarr_data = load_zarr_data_for_cell(zarr_url, cell_bounds, epsg_code)
        total_zarr_load_time += time.time() - zarr_load_start
        
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
        
        # Use the product_epsg from zarr_data (already resolved from metadata or inferred from tile ID)
        product_epsg = zarr_data.get('product_epsg')
        if product_epsg is None:
            # Skip this scene if we couldn't determine the CRS
            continue
        
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
        'zarr_load_time': total_zarr_load_time,
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
    
    # Track total processing time
    algorithm_start_time = time.time()
    
    # Reset global caches for each run
    global _failed_zarr_urls, _scene_cache, _stac_catalog
    _failed_zarr_urls = set()
    _scene_cache = {}
    _stac_catalog = None
    
    if verbose:
        print("=" * 80)
        print("GLACIER MONITORING ALGORITHM - SPATIAL EXPANSION")
        if low_memory:
            print("(LOW MEMORY MODE: Writing tiles to disk)")
        print("=" * 80)
    
    # Validate STAC connection before processing
    if verbose:
        print("\n[Step 0] Validating STAC connection...")
    if not validate_stac_connection(config['stac_url'], verbose):
        raise ConnectionError(f"Cannot connect to STAC catalog at {config['stac_url']}")
    
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
    
    # Initialize timing tracker
    global _cell_timing_tracker
    _cell_timing_tracker = CellTimingTracker()
    
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
        iteration_start = time.time()
        
        # Start iteration timing
        _cell_timing_tracker.start_iteration(iteration, len(unprocessed))
        
        # Create progress bar for cell processing
        pbar = tqdm(
            unprocessed.iterrows(),
            total=len(unprocessed),
            desc=f"Iter {iteration}",
            unit="cell",
            ncols=100,
            disable=not verbose,
            leave=True
        )
        
        for idx, cell_row in pbar:
            cell_id = cell_row['cell_id']
            cell_geom = cell_row.geometry
            cell_bounds = cell_geom.bounds
            
            # Start timing for this cell
            _cell_timing_tracker.start_cell(cell_id)
            
            # Update progress bar postfix
            pbar.set_postfix(cell=cell_id, snow=cells_with_snow, added=iteration_adds)
            
            # Query STAC for this cell (timed)
            stac_start = time.time()
            items = query_stac_for_cell(
                cell=cell_row,
                date_start=config['date_start'],
                date_end=config['date_end'],
                epsg_code=config['epsg_iceland'],
                stac_url=config['stac_url'],
                max_cloud_cover=config.get('max_cloud_cover', 50),
                verbose=False
            )
            stac_duration = time.time() - stac_start
            _cell_timing_tracker.record_stac_query(stac_duration, len(items))
            
            if len(items) == 0:
                grid_idx = grid[grid['cell_id'] == cell_id].index[0]
                grid.at[grid_idx, 'is_processed'] = True
                _cell_timing_tracker.end_cell(snow_percentage=None)
                continue
            
            # Compute NDSI for this cell (timed - includes Zarr loading)
            ndsi_start = time.time()
            result = compute_median_ndsi_for_cell(
                stac_items=items,
                cell_bounds=cell_bounds,
                epsg_code=config['epsg_iceland'],
                ndsi_threshold=config['ndsi_threshold'],
                max_scenes=config['max_scenes']
            )
            ndsi_duration = time.time() - ndsi_start
            
            if result is None:
                _cell_timing_tracker.record_ndsi_compute(ndsi_duration)
                grid_idx = grid[grid['cell_id'] == cell_id].index[0]
                grid.at[grid_idx, 'is_processed'] = True
                _cell_timing_tracker.end_cell(snow_percentage=None)
                gc.collect()  # Free memory
                continue
            
            # Record NDSI computation time (includes Zarr loading)
            _cell_timing_tracker.record_zarr_load(result.get('zarr_load_time', 0), result.get('scene_count', 0))
            _cell_timing_tracker.record_ndsi_compute(ndsi_duration - result.get('zarr_load_time', 0))
            
            # Store result - either in memory or on disk
            snow_pct = result['snow_percentage']
            cell_bounds_dict[cell_id] = cell_bounds
            
            if low_memory and tile_dir:
                # Save NDSI tile to disk immediately (timed)
                tile_save_start = time.time()
                tile_path = tile_dir / f"ndsi_cell_{cell_id}.tif"
                ndsi_da = result['ndsi_median']
                ndsi_da.rio.write_crs(f"EPSG:{config['epsg_iceland']}", inplace=True)
                ndsi_da.rio.to_raster(tile_path, driver='GTiff', compress='lzw')
                tile_save_duration = time.time() - tile_save_start
                _cell_timing_tracker.record_tile_save(tile_save_duration)
                del ndsi_da, result  # Free memory immediately
            else:
                # Store in memory (original behavior)
                ndsi_results[cell_id] = result['ndsi_median']
                grid_idx = grid[grid['cell_id'] == cell_id].index[0]
                grid.at[grid_idx, 'ndsi_median'] = result['ndsi_median']
            
            # Add snow percentage to grid
            grid_idx = grid[grid['cell_id'] == cell_id].index[0]
            grid.at[grid_idx, 'snow_percentage'] = snow_pct
            
            # Apply spatial expansion (timed)
            expansion_start = time.time()
            if snow_pct > 0:
                cells_with_snow += 1
                new_adds = apply_spatial_expansion(
                    grid, cell_id, snow_pct, threshold=config['snow_percentage_threshold']
                )
                iteration_adds += new_adds
            expansion_duration = time.time() - expansion_start
            _cell_timing_tracker.record_expansion(expansion_duration)
            
            # Mark as processed
            grid.at[grid_idx, 'is_processed'] = True
            total_processed += 1
            
            # End timing for this cell
            _cell_timing_tracker.end_cell(snow_percentage=snow_pct)
            
            # Log event to Proxmox monitor for correlation (if enabled)
            if _proxmox_monitor and _proxmox_monitor.enabled:
                cell_timing = _cell_timing_tracker.cell_timings[-1] if _cell_timing_tracker.cell_timings else {}
                _proxmox_monitor.add_event('cell_processed', {
                    'cell_id': cell_id,
                    'snow_percentage': snow_pct,
                    'stac_query_time': cell_timing.get('stac_query_time', 0),
                    'zarr_load_time': cell_timing.get('zarr_load_time', 0),
                    'ndsi_compute_time': cell_timing.get('ndsi_compute_time', 0),
                    'tile_save_time': cell_timing.get('tile_save_time', 0),
                    'total_time': cell_timing.get('total_time', 0)
                })
            
            # Force garbage collection to free memory
            gc.collect()
        
        # Close progress bar
        pbar.close()
        
        # Calculate iteration time
        iteration_time = time.time() - iteration_start
        
        # End iteration timing
        _cell_timing_tracker.end_iteration(len(unprocessed), cells_with_snow, iteration_adds)
        
        # Log iteration event to Proxmox monitor
        if _proxmox_monitor and _proxmox_monitor.enabled:
            _proxmox_monitor.add_event('iteration_completed', {
                'iteration': iteration,
                'cells_processed': len(unprocessed),
                'cells_with_snow': cells_with_snow,
                'new_candidates': iteration_adds,
                'iteration_time': iteration_time
            })
        
        total_expansion_adds += iteration_adds
        
        if verbose:
            print(f"   Summary: {len(unprocessed)} processed, {cells_with_snow} with snow, {iteration_adds} new candidates")
            print(f"   Iteration time: {iteration_time:.1f}s ({iteration_time/len(unprocessed):.2f}s/cell avg)")
    
    # Print summary of failed Zarr stores
    if verbose and len(_failed_zarr_urls) > 0:
        print(f"\n   ⚠ Warning: {len(_failed_zarr_urls)} unique Zarr stores failed to load")
        print(f"   This may indicate the data is not yet available on the server.")
        print(f"   Consider using a different date range or checking the EOPF catalog status.")
    
    # Step 3: Combine results or calculate statistics from tiles
    if low_memory:
        # In low-memory mode, we don't combine - just calculate stats from saved tiles
        if verbose:
            print(f"\n[Step 3] Calculating statistics (low-memory mode - reading tiles for accurate counts)...")
        
        # Read saved tiles to get accurate valid pixel counts (matching old script behavior)
        valid_pixels = 0
        snow_pixels = 0
        
        if tile_dir and tile_dir.exists():
            tile_files = list(tile_dir.glob("*.tif"))
            for tile_path in tile_files:
                try:
                    with rasterio.open(tile_path) as src:
                        data = src.read(1)
                        # Count valid (non-NaN) pixels
                        valid_mask = ~np.isnan(data)
                        tile_valid = int(valid_mask.sum())
                        # Count snow pixels (NDSI >= threshold)
                        tile_snow = int((data[valid_mask] >= config['ndsi_threshold']).sum())
                        valid_pixels += tile_valid
                        snow_pixels += tile_snow
                except Exception as e:
                    if verbose:
                        print(f"    Warning: Could not read tile {tile_path.name}: {e}")
        
        # Calculate overall snow percentage from actual pixel counts
        total_snow_pct = 100.0 * snow_pixels / valid_pixels if valid_pixels > 0 else 0.0
        
        # Calculate total processing time
        total_processing_time = time.time() - algorithm_start_time
        
        # Get timing summary
        timing_summary = _cell_timing_tracker.get_summary() if _cell_timing_tracker else {}
        
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
            'tile_directory': str(tile_dir) if tile_dir else None,
            'processing_time_seconds': round(total_processing_time, 2),
            'processing_time_formatted': f"{int(total_processing_time // 60)}m {int(total_processing_time % 60)}s",
            'cell_timing': timing_summary
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
        
        # Get timing summary
        timing_summary = _cell_timing_tracker.get_summary() if _cell_timing_tracker else {}
        
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
            'mode': 'normal',
            'processing_time_seconds': round(time.time() - algorithm_start_time, 2),
            'processing_time_formatted': f"{int((time.time() - algorithm_start_time) // 60)}m {int((time.time() - algorithm_start_time) % 60)}s",
            'cell_timing': timing_summary
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
        print(f"Processing time:              {statistics['processing_time_formatted']}")
        
        # Print timing breakdown if available
        if 'cell_timing' in statistics and statistics['cell_timing']:
            timing = statistics['cell_timing']
            print("\n" + "-" * 40)
            print("CELL PROCESSING TIME BREAKDOWN")
            print("-" * 40)
            if 'total_times' in timing:
                tt = timing['total_times']
                print(f"Total STAC query time:        {tt.get('stac_query_seconds', 0):.1f}s")
                print(f"Total Zarr load time:         {tt.get('zarr_load_seconds', 0):.1f}s")
                print(f"Total NDSI compute time:      {tt.get('ndsi_compute_seconds', 0):.1f}s")
                print(f"Total tile save time:         {tt.get('tile_save_seconds', 0):.1f}s")
                print(f"Total expansion time:         {tt.get('spatial_expansion_seconds', 0):.1f}s")
            if 'time_distribution_percent' in timing:
                td = timing['time_distribution_percent']
                print(f"\nTime distribution:")
                print(f"  STAC query:                 {td.get('stac_query', 0):.1f}%")
                print(f"  Zarr loading:               {td.get('zarr_load', 0):.1f}%")
                print(f"  NDSI computation:           {td.get('ndsi_compute', 0):.1f}%")
                print(f"  Tile saving:                {td.get('tile_save', 0):.1f}%")
                print(f"  Spatial expansion:          {td.get('spatial_expansion', 0):.1f}%")
            if 'average_times_per_cell' in timing:
                at = timing['average_times_per_cell']
                print(f"\nAverage time per cell:        {at.get('total_seconds', 0):.2f}s")
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
    parser.add_argument('--max-cloud-cover', type=int, default=DEFAULT_CONFIG['max_cloud_cover'],
                        help=f'Maximum cloud cover %% to include scenes (default: {DEFAULT_CONFIG["max_cloud_cover"]})')
    parser.add_argument('--low-memory', action='store_true', default=DEFAULT_CONFIG['low_memory'],
                        help='Enable low-memory mode: saves tiles to disk instead of RAM (recommended for large areas)')
    parser.add_argument('--no-low-memory', action='store_true',
                        help='Disable low-memory mode: keep all results in RAM')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')
    
    # Proxmox monitoring arguments
    parser.add_argument('--proxmox-monitor', action='store_true',
                        help='Enable Proxmox VM monitoring during processing')
    parser.add_argument('--proxmox-host', type=str, default=PROXMOX_CONFIG['host'],
                        help=f'Proxmox host IP (default: {PROXMOX_CONFIG["host"]})')
    parser.add_argument('--proxmox-node', type=str, default=PROXMOX_CONFIG['node'],
                        help=f'Proxmox node name (default: {PROXMOX_CONFIG["node"]})')
    parser.add_argument('--proxmox-vmid', type=int, default=PROXMOX_CONFIG['vmid'],
                        help=f'VM ID to monitor (default: {PROXMOX_CONFIG["vmid"]})')
    parser.add_argument('--proxmox-poll-interval', type=int, default=PROXMOX_CONFIG['poll_interval'],
                        help=f'Polling interval in seconds (default: {PROXMOX_CONFIG["poll_interval"]})')
    
    args = parser.parse_args()
    
    # Load configuration
    config = DEFAULT_CONFIG.copy()
    config['date_start'] = args.date_start
    config['date_end'] = args.date_end
    config['grid_size'] = args.grid_size
    config['ndsi_threshold'] = args.ndsi_threshold
    config['snow_percentage_threshold'] = args.snow_threshold
    config['max_scenes'] = args.max_scenes
    config['max_cloud_cover'] = args.max_cloud_cover
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
        print(f"Proxmox monitor:   {'ON' if args.proxmox_monitor else 'OFF'}")
        print("=" * 80)
    
    # Initialize Proxmox monitor if enabled
    proxmox_monitor = None
    if args.proxmox_monitor:
        proxmox_config = PROXMOX_CONFIG.copy()
        proxmox_config['enabled'] = True
        proxmox_config['host'] = args.proxmox_host
        proxmox_config['node'] = args.proxmox_node
        proxmox_config['vmid'] = args.proxmox_vmid
        proxmox_config['poll_interval'] = args.proxmox_poll_interval
        proxmox_monitor = ProxmoxMonitor(proxmox_config)
        if verbose:
            print(f"\nProxmox monitoring configured:")
            print(f"  Host: {args.proxmox_host}")
            print(f"  Node: {args.proxmox_node}")
            print(f"  VM ID: {args.proxmox_vmid}")
            print(f"  Poll interval: {args.proxmox_poll_interval}s")
    
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
    
    # Start Proxmox monitoring
    if proxmox_monitor:
        proxmox_monitor.start()
        proxmox_monitor.add_event('algorithm_start', {'seeds_count': len(seeds), 'grid_size': args.grid_size})
    
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
    
    # Stop Proxmox monitoring and get report
    proxmox_report = None
    if proxmox_monitor:
        proxmox_monitor.add_event('algorithm_end', {
            'cells_processed': results['statistics'].get('total_cells_processed', 0),
            'snow_percentage': results['statistics'].get('snow_ice_percentage', 0)
        })
        proxmox_report = proxmox_monitor.stop()
        
        # Save Proxmox report
        report_path = proxmox_monitor.save_report(output_dir, 'proxmox_monitoring')
        if verbose and report_path:
            print(f"\nProxmox monitoring report saved: {report_path}")
            print(f"  Samples collected: {proxmox_report.get('samples_collected', 0)}")
            print(f"  Network IN: {proxmox_report.get('network_transfer', {}).get('mb_in', 0)} MB")
            print(f"  Network OUT: {proxmox_report.get('network_transfer', {}).get('mb_out', 0)} MB")
            print(f"  Avg CPU: {proxmox_report.get('cpu_stats', {}).get('avg_percent', 0)}%")
            print(f"  Avg Memory: {proxmox_report.get('memory_stats', {}).get('avg_percent', 0)}%")
    
    # Export results
    try:
        export_results(results, output_dir, config)
    
    except Exception as e:
        print(f"ERROR: Failed to export results: {e}")
        if proxmox_monitor:
            proxmox_monitor.stop()  # Ensure monitoring stops on error
        import traceback
        traceback.print_exc()
        return 1
    
    # Display all configuration settings
    if verbose:
        print("\n" + "=" * 60)
        print("CONFIGURATION SETTINGS USED")
        print("=" * 60)
        for key, value in sorted(config.items()):
            print(f"  {key}: {value}")
        print("=" * 60)
    
    if verbose:
        print("\nGlacier monitoring completed successfully!")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
