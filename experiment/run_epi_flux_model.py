import numpy as np
import pandas as pd
import geopandas as gpd
import h5py
import os
import sys
import tarfile
sys.path.append(os.path.abspath(
    os.path.join('__file__', '..')
))
from util.file_util import check_exist

########################
##### parameter settings
########################
extract_open_census = 0    # set to 1 if SafeGraph Open Census Data is not extracted.

###################
##### File settings
###################
# data directory
data_dir = os.path.abspath(
    os.path.join(os.path.curdir, 'data')
)
check_exist(data_dir, 'dir')

# result directory
res_dir = os.path.abspath(
    os.path.join(os.path.curdir, 'results')
)
check_exist(res_dir, 'dir')

# open result file 
resfile = os.path.join(
    res_dir, 'safegraph_analysis.hdf5'
)
check_exist(resfile, 'file')

complevel=7
complib='zlib'
with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
    print(f"File {resfile} has {len(store.keys())} entries.")

##########################################################
##### Extract SafeGraph Open Census Data if it is not yet.
##########################################################
if extract_open_census:
    # open SafeGraph Open Census Data
    open_census_file = os.path.join(
        data_dir, 'safegraph_open_census_data.tgz'
    )
    check_exist(open_census_file, 'file')

    with tarfile.open(open_census_file, 'r') as tar:
        tar.extractall(path=data_dir)

########################
##### read location file
########################
geofile = os.path.join(
    data_dir, 'safegraph_open_census_data', 'geometry', 'cbg.geojson'
)
check_exist(geofile, 'file')

geo = gpd.read_file(geofile).astype({'CensusBlockGroup': 'int64'})
geo.set_index('CensusBlockGroup', inplace=True)
print(geo)