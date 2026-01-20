"""
Procesamiento de Manzanas - Entidades 01 a 08
Este script procesa imágenes satelitales para todas las entidades desde 01 hasta 05.
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
from math import sqrt
from datetime import datetime
import logging

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

import rasterio
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.warp import transform_geom

from google.cloud import storage
from tqdm import tqdm

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('procesamiento_01_08.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =========================
# CONFIGURACIÓN
# =========================

BUCKET_NAME = "ai360_datalake"
GCS_MANZANAS_PATH = "manzana/imagenes_satelitales/0. raw/manzanas_nacional.geoparquet"
GCS_IMAGES_PREFIX = "manzana/imagenes_satelitales/imagenes"
GCS_OUTPUT_PREFIX = "manzana/imagenes_satelitales/imagen_manzana"

# Entidades a procesar
ENTIDADES = [f"{i:02d}" for i in range(1, 9)]  # 01 a 08

LOCAL_TEMP_DIR = Path("./temp_processing")
LOCAL_TEMP_DIR.mkdir(exist_ok=True)

TARGET_CRS = "EPSG:3857"

# CSV para tracking
CSV_OUTPUT = Path("./procesadas_01_08.csv")

logger.info(f"Procesando entidades: {', '.join(ENTIDADES)}")

# =========================
# FUNCIONES
# =========================

def get_buffer_multiplier(area_m2):
    if area_m2 <= 5000:
        return 8.0
    elif area_m2 <= 10000:
        return 4.0
    elif area_m2 <= 20000:
        return 2.0
    else:
        return 1.0

def get_size_class(area_m2):
    if area_m2 <= 5000:
        return "muy_pequena"
    elif area_m2 <= 10000:
        return "pequena"
    elif area_m2 <= 20000:
        return "mediana"
    else:
        return "grande"

def calculate_buffer_distance(geometry, buffer_multiplier, area_m2=None):
    if area_m2 is None:
        area_m2 = geometry.area
    
    if area_m2 < 100:
        return 20
    
    target_area = area_m2 * buffer_multiplier
    original_radius = sqrt(area_m2 / np.pi)
    target_radius = sqrt(target_area / np.pi)
    buffer_distance = target_radius - original_radius
    
    if area_m2 <= 5000:
        buffer_distance = max(min(buffer_distance, 100), 20)
    elif area_m2 <= 10000:
        buffer_distance = max(min(buffer_distance, 80), 15)
    elif area_m2 <= 20000:
        buffer_distance = max(min(buffer_distance, 60), 10)
    else:
        buffer_distance = max(min(buffer_distance, 40), 5)
    
    return buffer_distance

def download_blob_to_tempfile(bucket, blob_path, local_path):
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)
    return local_path

def upload_file_to_gcs(bucket, local_path, gcs_path):
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    return f"gs://{BUCKET_NAME}/{gcs_path}"

def list_blobs_with_prefix(bucket, prefix):
    blobs = bucket.list_blobs(prefix=prefix)
    return [blob.name for blob in blobs]

def find_intersecting_images(geometry, images_gdf):
    temp_gdf = gpd.GeoDataFrame([{'geometry': geometry}], crs=images_gdf.crs)
    intersecting = gpd.sjoin(images_gdf, temp_gdf, how='inner', predicate='intersects')
    return intersecting

def create_temp_raster(data, transform, meta, temp_dir):
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = tempfile.NamedTemporaryFile(dir=temp_dir, suffix='.tif', delete=False)
    temp_path = temp_file.name
    temp_file.close()
    
    with rasterio.open(temp_path, 'w', **meta) as dst:
        dst.write(data)
    
    return temp_path

def crop_manzana_image(manzana_geom, manzana_id, area_m2, buffer_multiplier, cve_ent, cvegeo, 
                       images_gdf, manzanas_filtradas, bucket):
    try:
        buffer_distance = calculate_buffer_distance(manzana_geom, buffer_multiplier, area_m2)
        buffered_geom = manzana_geom.buffer(buffer_distance)
        
        intersecting_imgs = find_intersecting_images(buffered_geom, images_gdf)
        
        if len(intersecting_imgs) == 0:
            return False
        
        if len(intersecting_imgs) == 1:
            img_path = intersecting_imgs.iloc[0]['local_path']
            with rasterio.open(img_path) as src:
                if str(manzanas_filtradas.crs) != str(src.crs):
                    buffered_geom_proj = transform_geom(
                        str(manzanas_filtradas.crs),
                        str(src.crs),
                        buffered_geom.__geo_interface__
                    )
                    geoms = [buffered_geom_proj]
                else:
                    geoms = [buffered_geom.__geo_interface__]
                
                out_image, out_transform = mask(src, geoms, crop=True, filled=True, nodata=0)
                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })
        else:
            src_files = [rasterio.open(row['local_path']) for _, row in intersecting_imgs.iterrows()]
            
            try:
                mosaic, out_trans = merge(src_files, bounds=buffered_geom.bounds)
                meta = src_files[0].meta.copy()
                meta.update({
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_trans
                })
                
                temp_mosaic_path = create_temp_raster(mosaic, out_trans, meta, LOCAL_TEMP_DIR / "mosaics")
                
                with rasterio.open(temp_mosaic_path) as src_mosaic:
                    if str(manzanas_filtradas.crs) != str(src_mosaic.crs):
                        buffered_geom_proj = transform_geom(
                            str(manzanas_filtradas.crs),
                            str(src_mosaic.crs),
                            buffered_geom.__geo_interface__
                        )
                        geoms = [buffered_geom_proj]
                    else:
                        geoms = [buffered_geom.__geo_interface__]
                    
                    out_image, out_transform = mask(src_mosaic, geoms, crop=True, filled=True, nodata=0)
                    out_meta = src_mosaic.meta.copy()
                    out_meta.update({
                        "driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform
                    })
                
                os.unlink(temp_mosaic_path)
            finally:
                for src in src_files:
                    src.close()
        
        # Validar que out_image es válido
        if out_image is None or out_image.size == 0:
            logger.error(f"out_image inválido para {manzana_id}")
            return False
        
        # Asegurar que out_meta tiene todos los campos requeridos
        if 'height' not in out_meta or 'width' not in out_meta:
            logger.error(f"out_meta incompleto para {manzana_id}: {out_meta.keys()}")
            return False
        
        try:
            local_tif_path = LOCAL_TEMP_DIR / f"manzana_{manzana_id}.tif"
            
            # Asegurar que el directorio existe
            local_tif_path.parent.mkdir(parents=True, exist_ok=True)
            
            with rasterio.open(local_tif_path, "w", **out_meta) as dest:
                dest.write(out_image)
            
            # Verificar que el archivo se escribió correctamente
            if not local_tif_path.exists() or local_tif_path.stat().st_size == 0:
                logger.error(f"Archivo {manzana_id} no se escribió correctamente")
                return False
            
            gcs_tif_path = f"{GCS_OUTPUT_PREFIX}/{cve_ent}/{cvegeo}.tif"
            upload_file_to_gcs(bucket, str(local_tif_path), gcs_tif_path)
            local_tif_path.unlink()
            
            return True
        except Exception as e:
            logger.error(f"Error escribiendo archivo {manzana_id}: {str(e)}")
            if local_tif_path.exists():
                try:
                    local_tif_path.unlink()
                except:
                    pass
            return False
        
    except Exception as e:
        logger.error(f"Error en {manzana_id}: {str(e)}")
        return False

def process_entidad(cve_ent, storage_client, bucket, manzanas_all):
    logger.info(f"\n{'='*60}")
    logger.info(f"PROCESANDO ENTIDAD: {cve_ent}")
    logger.info(f"{'='*60}")
    
    # Filtrar manzanas
    manzanas_filtradas = manzanas_all[manzanas_all['cve_ent'] == cve_ent].copy()
    manzanas_filtradas = manzanas_filtradas[manzanas_filtradas['ambito'] == "Urbana"].copy()
    
    if len(manzanas_filtradas) == 0:
        logger.warning(f"No hay manzanas urbanas para entidad {cve_ent}")
        return []
    
    logger.info(f"Manzanas urbanas encontradas: {len(manzanas_filtradas):,}")
    
    # Preparar datos
    manzanas_filtradas = manzanas_filtradas.to_crs(TARGET_CRS)
    manzanas_filtradas['area_m2'] = manzanas_filtradas.geometry.area
    manzanas_filtradas['buffer_multiplier'] = manzanas_filtradas['area_m2'].apply(get_buffer_multiplier)
    manzanas_filtradas['size_class'] = manzanas_filtradas['area_m2'].apply(get_size_class)
    
    # Cargar imágenes
    images_prefix = f"{GCS_IMAGES_PREFIX}/{cve_ent}/"
    logger.info(f"Buscando imágenes en: gs://{BUCKET_NAME}/{images_prefix}")
    
    image_blobs = [blob for blob in list_blobs_with_prefix(bucket, images_prefix) if blob.endswith('.tif')]
    
    if len(image_blobs) == 0:
        logger.warning(f"No se encontraron imágenes para entidad {cve_ent}")
        return []
    
    logger.info(f"Total imágenes encontradas: {len(image_blobs):,}")
    
    # Descargar y crear índice espacial
    local_images_dir = LOCAL_TEMP_DIR / f"images_{cve_ent}"
    local_images_dir.mkdir(exist_ok=True)
    
    image_extents = []
    logger.info("Descargando metadatos de imágenes...")
    
    for blob_path in tqdm(image_blobs, desc=f"Cargando metadatos {cve_ent}"):
        try:
            local_img_path = local_images_dir / Path(blob_path).name
            
            if not local_img_path.exists():
                download_blob_to_tempfile(bucket, blob_path, str(local_img_path))
            
            with rasterio.open(local_img_path) as src:
                bounds = src.bounds
                geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
                temp_gdf = gpd.GeoDataFrame([{'geometry': geom}], crs=src.crs)
                geom_transformed = temp_gdf.to_crs(TARGET_CRS).geometry.iloc[0]
                image_extents.append({
                    'blob_path': blob_path,
                    'local_path': str(local_img_path),
                    'geometry': geom_transformed,
                    'crs': src.crs
                })
        except Exception as e:
            logger.error(f"Error con {blob_path}: {e}")
    
    images_gdf = gpd.GeoDataFrame(image_extents, crs=TARGET_CRS)
    logger.info(f"Índice espacial creado con {len(images_gdf):,} imágenes")
    
    # Procesar manzanas
    stats = {
        'muy_pequena': {'total': 0, 'success': 0, 'error': 0},
        'pequena': {'total': 0, 'success': 0, 'error': 0},
        'mediana': {'total': 0, 'success': 0, 'error': 0},
        'grande': {'total': 0, 'success': 0, 'error': 0}
    }
    
    processed_records = []
    
    logger.info("Iniciando procesamiento de manzanas...")
    for idx, row in tqdm(manzanas_filtradas.iterrows(), total=len(manzanas_filtradas), 
                         desc=f"Procesando {cve_ent}"):
        cvegeo = row['cvegeo']
        geometry = row['geometry']
        area_m2 = row['area_m2']
        buffer_multiplier = row['buffer_multiplier']
        size_class = row['size_class']
        
        stats[size_class]['total'] += 1
        
        # Verificar si ya existe
        gcs_tif_path = f"{GCS_OUTPUT_PREFIX}/{cve_ent}/{cvegeo}.tif"
        blob = bucket.blob(gcs_tif_path)
        
        if blob.exists():
            stats[size_class]['success'] += 1
            processed_records.append({
                'cvegeo': cvegeo,
                'cve_ent': cve_ent,
                'area_m2': area_m2,
                'size_class': size_class,
                'status': 'ya_existe',
                'timestamp': datetime.now().isoformat()
            })
            continue
        
        # Procesar
        success = crop_manzana_image(
            geometry, cvegeo, area_m2, buffer_multiplier, cve_ent, cvegeo,
            images_gdf, manzanas_filtradas, bucket
        )
        
        if success:
            stats[size_class]['success'] += 1
            processed_records.append({
                'cvegeo': cvegeo,
                'cve_ent': cve_ent,
                'area_m2': area_m2,
                'size_class': size_class,
                'status': 'procesada',
                'timestamp': datetime.now().isoformat()
            })
        else:
            stats[size_class]['error'] += 1
            processed_records.append({
                'cvegeo': cvegeo,
                'cve_ent': cve_ent,
                'area_m2': area_m2,
                'size_class': size_class,
                'status': 'sin_imagenes',
                'timestamp': datetime.now().isoformat()
            })
    
    # Resumen
    total_success = sum(s['success'] for s in stats.values())
    total_error = sum(s['error'] for s in stats.values())
    
    logger.info(f"\nRESUMEN ENTIDAD {cve_ent}:")
    logger.info(f"  Total: {len(manzanas_filtradas):,}")
    logger.info(f"  Éxito: {total_success:,}")
    logger.info(f"  Error: {total_error:,}")
    
    # Limpiar imágenes de esta entidad
    if local_images_dir.exists():
        shutil.rmtree(local_images_dir)
    
    return processed_records

def main():
    logger.info("="*60)
    logger.info("INICIANDO PROCESAMIENTO MASIVO - ENTIDADES 01 a 08")
    logger.info("="*60)
    
    # Inicializar GCS
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    
    # Cargar manzanas
    logger.info(f"Descargando manzanas desde GCS: {GCS_MANZANAS_PATH}")
    local_manzanas_path = LOCAL_TEMP_DIR / "manzanas_nacional.geoparquet"
    
    if not local_manzanas_path.exists():
        download_blob_to_tempfile(bucket, GCS_MANZANAS_PATH, str(local_manzanas_path))
    
    manzanas_all = gpd.read_parquet(local_manzanas_path)
    logger.info(f"Total manzanas cargadas: {len(manzanas_all):,}")
    
    # Procesar cada entidad
    all_processed = []
    
    for cve_ent in ENTIDADES:
        try:
            records = process_entidad(cve_ent, storage_client, bucket, manzanas_all)
            all_processed.extend(records)
            
            # Guardar CSV intermedio
            if all_processed:
                df_temp = pd.DataFrame(all_processed)
                df_temp.to_csv(CSV_OUTPUT, index=False)
                logger.info(f"CSV actualizado: {len(all_processed)} registros")
        
        except Exception as e:
            logger.error(f"Error procesando entidad {cve_ent}: {str(e)}")
            continue
    
    # Guardar CSV final
    if all_processed:
        df_final = pd.DataFrame(all_processed)
        df_final.to_csv(CSV_OUTPUT, index=False)
        logger.info(f"\nCSV FINAL guardado: {CSV_OUTPUT}")
        logger.info(f"Total registros: {len(all_processed):,}")
        
        # Resumen por estado
        summary = df_final.groupby(['cve_ent', 'status']).size().unstack(fill_value=0)
        logger.info("\nRESUMEN POR ENTIDAD:")
        logger.info(f"\n{summary}")
    
    logger.info("\n" + "="*60)
    logger.info("PROCESO COMPLETADO")
    logger.info("="*60)

if __name__ == "__main__":
    main()
