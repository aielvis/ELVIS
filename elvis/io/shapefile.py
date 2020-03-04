""" Convenience functions for loading shapefile into Jupyter
"""
from os.path import join

import geopandas as gpd
import numpy as np
import pandas as pd
import shapefile


def contour_to_geojson(base_directory, file_prefix, correct_bbox=True):
    gom_shape = join(base_directory, file_prefix + ".shp")
    gom_dbf = join(base_directory, file_prefix + ".dbf")
    gom_shx = join(base_directory, file_prefix + ".shx")

    reader = shapefile.Reader(gom_shape)
    shp = reader.load_dbf(gom_dbf)
    shx = reader.load_shx(gom_shx)

    contour_json = reader.__geo_interface__

    if bbox in contour_json:
        # corrects json issue    
        contour_json['bbox'] = tuple(contour_json['bbox'])

    return contour_json


def boem_lease_owner(base_directory):
    """ The reason for this function is to set the various fields to 
        to correct data-types on load."""
    
    # this data only goes back to 2000
    boem_lease_owner_dtypes = {"MMS Company Number" : int,
                               "Assignment Pct" : float}
    
    boem_lease_owner_dates = ["Approved Date", 
                              "MMS Start Date", 
                              "Asgn Eff Date"]
    # block numbers can contain a character e.g. "159 A"
    boem_lease_owner = pd.read_csv(join(base_directory, 
                                        "LeaseOwner.csv"),
                                   dtype=boem_lease_owner_dtypes,
                                   parse_dates=boem_lease_owner_dates)

    boem_lease_owner["Company Name"] = \
                            boem_lease_owner["Company Name"].str.lower()
    
    # some leases are owned by a consortia, use the largest partner
    boem_lease_owner.sort_values(["Lease Number", "Assignment Pct"], 
                                 inplace=True)
    # label where 
    boem_lease_owner["Consortia"] = boem_lease_owner.duplicated(
        subset=["Lease Number"]
    )
    
    # keep a record of the consortia 
    consortia_leases = boem_lease_owner[
        boem_lease_owner.duplicated(subset=["Lease Number"])].copy()
    consortia_leases.set_index("Lease Number", inplace=True)
    
    # use the largest owner to index
    boem_lease_owner.drop_duplicates(subset=["Lease Number"], keep="last", 
                                     inplace=True)
    boem_lease_owner.set_index("Lease Number", inplace=True)

    return boem_lease_owner, consortia_leases

def boem_leases(base_directory):
    """ The reason for this function is to set the various fields to 
        to correct data-types on load."""

    boem_lease_dtypes = {"Block Max Water Depth (meters)" : int}
    boem_lease_dates = ["Lease Effective Date", "Lease Expiration Date"]
    # block numbers can contain a character e.g. "159 A"
    boem_leases = pd.read_csv(join(base_directory, "LeaseAreaBlock.csv"),
                              dtype=boem_lease_dtypes,
                              parse_dates=boem_lease_dates)
    boem_leases.set_index("Lease Number", inplace=True)
    
    # block numbers aren't ints 
    boem_leases["Block Number"] = \
                        boem_leases["Block Number"].str.replace(' +', '')
    # AREABLK same as AC_BLK in shape files
    # FIXME - check codes like AT001 <-> AT1
    boem_leases["AREABLK"] = boem_leases["Area Code"] + \
                             boem_leases["Block Number"]

    return boem_leases

def boem_lease_by_owner(base_directory, owner=None):
    """ Supply an owner e.g. "equinor" if you want to have a single owner, 
        otherwise the data will contain all owned leases."""
    
    lease_owners, consortia = boem_lease_owner(base_directory)
    leases = boem_leases(base_directory)
    
    # combines the lease information with the owner
    owned_leases = leases.join(lease_owners, how='inner')

    if owner:
        # subset
        _leases = owned_leases["Company Name"].str.contains(owner)
        owned_leases = owned_leases[_leases].copy()

    return owned_leases, consortia

def get_current_leases(owned_leases, inplace=False):
    # find the current ones
    indx = np.logical_or(
        owned_leases["Lease Expiration Date"] < pd.Timestamp.now(),
        owned_leases["Lease Expiration Date"].isnull())

    if inplace:
        owned_leases = owned_leases[indx]
        return owned_leases
    else:    
        return owned_leases[indx].copy()

def get_blocks(base_directory):
    block_directory = join(base_directory, "blocks")
    block_shape = join(block_directory, "blocks.shp")

    return gpd.read_file(block_shape)

def get_blocks_by_owner(base_directory, owner=None):
    """ Supply an owner e.g. "equinor" if you want to have a single owner, 
        otherwise the data will contain all owned leases."""
    
    owned_leases = boem_lease_by_owner(base_directory, owner=owner)
    blocks = get_blocks(base_directory)
    
    # combine equinor's current leases with shapefile
    owned_leases_by_block = blocks.merge(owned_leases, 
                                         how="inner", 
                                         left_on='AC_LAB', 
                                         right_on='AREABLK')
    return owned_leases_by_block
