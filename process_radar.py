import os
import shutil
import subprocess
import glob
import json
import logging
from datetime import datetime, timezone, timedelta
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import warnings
import re

# Suppress warnings
warnings.filterwarnings("ignore")

# Config
RCLONE_REMOTE = "gdrive" # Need to configure rclone.conf via secrets
SOURCE_PATH = os.environ.get("RADAR_SOURCE_PATH", "cart_no_clutter") # Configurable via Env Var
WORK_DIR = "/app/work"
OUTPUT_DIR = "/app/output" # This will be deployed to gh-pages
MDV_DIR = os.path.join(WORK_DIR, "mdv")
NC_DIR = os.path.join(WORK_DIR, "nc")
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
ROLLING_WINDOW_SIZE = 5

# Ensure dirs
for d in [MDV_DIR, NC_DIR, IMAGES_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_rclone():
    """Configures rclone from env vars if needed"""
    # In GitHub Actions, we can just use the installed rclone and expect config 
    # OR pass config via env var `RCLONE_CONFIG_GDRIVE_TYPE`, etc.
    # Usually easier to write a rclone.conf from a secret.
    pass

def download_mdv():
    """Download new MDV files"""
    logging.info("Downloading MDV files...")
    # Select files from the last 2 hours to ensure we cover gaps but don't download history
    # This assumes the cron runs frequently. 
    # We might need to look back slightly more if there are delays.
    cmd = ["rclone", "copy", f"{RCLONE_REMOTE}:{SOURCE_PATH}", MDV_DIR, 
           "--include", "*.mdv", 
           "--max-age", "2h",
           "--verbose"]
    subprocess.run(cmd, check=True)

def convert_mdv_to_nc(mdv_path):
    """Convert MDV to NetCDF using Mdv2NetCDF"""
    filename = os.path.basename(mdv_path)
    base_name = os.path.splitext(filename)[0]
    nc_out_path = os.path.join(NC_DIR, f"{base_name}.nc")
    
    if os.path.exists(nc_out_path):
        return nc_out_path # Already converted

    # Create params file on the fly or use default
    # Mdv2NetCDF -f <file> -outDir <dir> -nc3
    cmd = ["Mdv2NetCDF", "-f", mdv_path, "-outDir", NC_DIR, "-nc3"]
    logging.info(f"Converting {filename} to NC...")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        # Verify output (Mdv2NetCDF output naming might add 'ncfdata' prefix or similar)
        # Check generated file
        generated = glob.glob(os.path.join(NC_DIR, f"*{base_name}*.nc"))
        if generated:
            return generated[0]
        else:
            logging.error(f"No NC file generated for {filename}")
            return None
    except subprocess.CalledProcessError as e:
        logging.error(f"Conversion failed for {filename}: {e.stderr.decode()}")
        return None

def generate_image(nc_path, output_path):
    """Generate transparent PNG and return bounds"""
    if os.path.exists(output_path):
         # Load cached bounds if json exists
         json_path = output_path + ".json"
         if os.path.exists(json_path):
             with open(json_path, 'r') as f:
                 return json.load(f)['bounds']

    logging.info(f"Generating image for {os.path.basename(nc_path)}")
    try:
        ds = xr.open_dataset(nc_path, decode_times=False)
        # Logic from pipeline_worker.py:
        # 1. Variable selection
        var_name = "DBZ" if "DBZ" in ds else list(ds.data_vars)[0]
        dbz_data = ds[var_name].squeeze().values
        
        # 2. Composite (Max over Z)
        # Skip first few levels if needed (skip_levels=2 in user code)
        skip_levels = 2
        if dbz_data.ndim == 3 and dbz_data.shape[0] > skip_levels:
             composite_data = np.nanmax(dbz_data[skip_levels:, :, :], axis=0)
        elif dbz_data.ndim == 3:
             composite_data = np.nanmax(dbz_data, axis=0)
        else:
             composite_data = dbz_data
             
        # 3. Projection
        proj_info = ds['grid_mapping_0'].attrs
        lon_0 = proj_info['longitude_of_projection_origin']
        lat_0 = proj_info['latitude_of_projection_origin']
        projection = ccrs.AzimuthalEquidistant(central_longitude=lon_0, central_latitude=lat_0)
        
        x = ds['x0'].values if 'x0' in ds else ds['longitude'].values
        y = ds['y0'].values if 'y0' in ds else ds['latitude'].values
        
        # 4. Plot
        fig = plt.figure(figsize=(10, 10), dpi=150)
        ax = fig.add_subplot(1, 1, 1, projection=projection) # Important: set projection?
        # Note: In worker code: ax = fig.add_subplot(1, 1, 1); fig.patch.set_alpha(0)...
        # But here we want a map? No, the worker code does NOT add map features (coastlines, etc), 
        # it just plots the array and saves it transparently to overlay on MapLibre.
        # So we just plot the image with extent.
        
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        ax.set_axis_off()
        
        # Use imshow with origin lower
        ax.imshow(composite_data, cmap='jet', origin='lower', vmin=0, vmax=70) # bounds are implicit in pixels? 
        # Wait, simply saving the plot works if we don't have axis. 
        # But MapLibre needs georeferenced coords.
        
        plt.tight_layout(pad=0)
        plt.savefig(output_path, dpi=150, transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
        # 5. Calculate Bounds (Lat/Lon)
        x_min, x_max = x.min() * 1000, x.max() * 1000
        y_min, y_max = y.min() * 1000, y.max() * 1000
        
        geo_proj = ccrs.Geodetic()
        sw_corner = geo_proj.transform_point(x_min, y_min, projection)
        ne_corner = geo_proj.transform_point(x_max, y_max, projection)
        
        # Format: [[lon_min, lat_min], [lon_max, lat_max]] ?? 
        # MapLibre expects [[West, South], [East, North]] i.e. [[minLon, minLat], [maxLon, maxLat]]
        # User worker returned: [[float(sw_corner[1]), float(sw_corner[0])], [float(ne_corner[1]), float(ne_corner[0])]] 
        # => [[lat_min, lon_min], [lat_max, lon_max]] 
        # WAIT, worker comment says: # Formato: [[lat_min, lon_min], [lat_max, lon_max]]
        # But my frontend code uses: 
        # bounds are [[lat_min, lon_min]...] or [[lon, lat]]?
        # Let's check visualizer code again.
        # getImageCoordinates in frontend: 
        # p1=b[0], p2=b[1]. minLat = min(p1[0], p2[0]) -> so index 0 is lat.
        # minLon = min(p1[1], p2[1]) -> index 1 is lon.
        # return coordinates for source: [[minLon, maxLat], [maxLon, maxLat], [maxLon, minLat], [minLon, minLat]]
        # So yes, python output [[val1, val2], [val3, val4]] where val1=lat, val2=lon is consistent.
        
        bounds = [[float(sw_corner[1]), float(sw_corner[0])], [float(ne_corner[1]), float(ne_corner[0])]]
        
        with open(output_path + ".json", 'w') as f:
            json.dump({"bounds": bounds}, f)
            
        return bounds

    except Exception as e:
        logging.error(f"Error generating image: {e}")
        return None

def main():
    setup_rclone()
    download_mdv()
    
    # Debug: List what rclone downloaded
    logging.info(f"Listing MDV_DIR ({MDV_DIR}) contents:")
    for root, dirs, files in os.walk(MDV_DIR):
        for name in files:
            logging.info(os.path.join(root, name))

    # Manual walk to find files if glob fails unpredictably
    mdv_files = []
    for root, dirs, files in os.walk(MDV_DIR):
        for name in files:
            if name.endswith(".mdv"):
                mdv_files.append(os.path.join(root, name))
    
    logging.info(f"Found {len(mdv_files)} MDV files via os.walk")
    
    # Parse timestamps for sorting
    # Filename format expected: 20260130_200000.mdv
    file_list = []
    for f in mdv_files:
        basename = os.path.basename(f)
        try:
            # Extract timestamp YYYYMMDD_HHMMSS
            # Adjust regex if filename has prefix/suffix
            match = re.search(r'(\d{8}_\d{6})', basename)
            if match:
                ts_str = match.group(1)
                dt = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
                file_list.append({'path': f, 'dt': dt, 'basename': basename})
        except ValueError:
            logging.warning(f"Skipping file with unknown format: {basename}")
            
    # Sort by datetime
    file_list.sort(key=lambda x: x['dt'])
    
    # Keep only the last N files (Rolling Window)
    if not file_list:
        logging.info("No MDV files found.")
        return

    # Select the target files
    target_files = file_list[-ROLLING_WINDOW_SIZE:]
    logging.info(f"Processing latest {len(target_files)} files: {[x['basename'] for x in target_files]}")
    
    data_list = []
    
    # Clean up output directory (remove old images not in target list)
    # This prevents the output folder from growing indefinitely
    # and ensures users only see what's in data.json (though frontend usually dictates that)
    # But clean disk is good.
    existing_images = glob.glob(os.path.join(IMAGES_DIR, "*.png"))
    target_image_names = [f"{x['basename'].replace('.mdv', '')}.png" for x in target_files]
    
    for img_path in existing_images:
        if os.path.basename(img_path) not in target_image_names:
            logging.info(f"Cleaning up old image: {os.path.basename(img_path)}")
            os.remove(img_path)
            if os.path.exists(img_path + ".json"):
                os.remove(img_path + ".json")

    for item in target_files:
        mdv = item['path']
        nc = convert_mdv_to_nc(mdv)
        if nc:
            basename = item['basename'].replace(".mdv", "")
            img_filename = f"{basename}.png"
            img_path = os.path.join(IMAGES_DIR, img_filename)
            
            bounds = generate_image(nc, img_path)
            
            if bounds:
                # Add to data list
                data_list.append({
                    "url": f"images/{img_filename}",
                    "bounds": bounds,
                    "target_time": basename
                })
    
    # Write data.json
    with open(os.path.join(OUTPUT_DIR, "data.json"), 'w') as f:
        json.dump(data_list, f, indent=2)
        
    logging.info("Processing complete.")

if __name__ == "__main__":
    main()
