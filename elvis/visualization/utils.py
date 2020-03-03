
from os.path import dirname, join

import geopandas as gpd
import numpy as np
import pandas as pd
import fiona
from sklearn.neighbors import KernelDensity, KDTree

from elvis import datasets
from elvis.io.boem_from_file import (boem_leases, 
                                     load_well_headers, 
                                     get_lease_features, 
                                     boem_lease_owner,
                                     provide_current,
                                     past_leases,
                                     get_blocks,
                                     get_neighbourhood_leases,
                                     read_curated_neighbourhoods,
                                     freeze_bids,
                                     get_prod)


from elvis.io.boem_from_file import (boem_platform_structures,
                                     get_pipelines_with_meta,
                                     platform_to_geo_json,
                                     platform_by_structure)

from elvis.visualization.mapping import (bathymetry_underlay,
                                         create_map_from_geojson,
                                         colors as company_colors,
                                         geojson_underlay)

base_directory = dirname(datasets.__file__)
freeze_data = join(base_directory, 'Freeze_Data\ 12_4_2019')

period_size = 'Q'
# date range for backtesting
date_range = pd.date_range(start=pd.datetime(2003,1,1), 
                           end=pd.datetime(2020,1,1), 
                           freq=period_size)
periods = date_range.to_period(period_size)


null_header = [{'AREABLK': 'None',
               'Lease Number': 'None',
               'Company Name': 'Unknown',
               'Lease Code': 'NA',
               'Lease Effective Date': pd.NaT,
               'Lease Expiration Date': pd.NaT,
               'Oil Production': 0,
               'Gas Production': 0,
               'Water Depth': 0,
               'Producing Completions': 0,
               'Consortia': 0,
               'Previous Interest': 0,
               'Min Bid': 0,
               'Max Bid': 0,
               'MROV': 0,
               'Max Water Depth': 0}]

null_past = [{'AREABLK': 'None',
             'Lease Number': 'None',
             'Company Name': 'Unknown',
             'Lease Code': 'NA',
             'Lease Effective Date': pd.NaT,
             'Lease Expiration Date': pd.NaT,
             'Oil Production': 0,
             'Gas Production': 0,
             'Water Depth': 0,
             'Producing Completions': 0,
             'Consortia': 0,
             'Previous Interest': 0,
             'Min Bid': 0,
             'Max Bid': 0,
             'MROV': 0,
             'Max Water Depth': 0}]

null_nn = [{'Lease Status Code': '',
           'Lease Effective Date': pd.NaT,
           'Lease Expiration Date': pd.NaT,
           'Block Max Water Depth (meters)': 0}]

def init_block_data(nn):
    nn["Company Name"] = "Unknown"
    nn["Consortia"] = 0
    nn["Previous Interest"] = 0
    nn["Min Bid"] = 0
    nn["Max Bid"] = 0
    nn["MROV"] = 0

    nn["Oil Production"] = 0
    nn["Gas Production"] = 0
    nn["Max Water Depth"] = 0
    nn["Producing Completions"] = 0
    
def combine_block_data(nn, key, leases,
                       qdata, bid_data, winning_bids, consortia):

    
    if key[1] == "None":
        return

    if not np.any(winning_bids.index.isin([key])):
        return

    
    most_recent, oil_prod, gas_prod, water_depth, num_compl = get_prod(
        key[1], qdata, cumulative=True)

    company_name = winning_bids.loc[key, "Company Name"]
    nn.loc[key, "Company Name"] = company_name
    nn.loc[key, "Consortia"] = int(winning_bids.loc[key, "Consortia"])
    
    # find previous interest
    _leases = leases.loc[key[0]]
    eff_date = leases.loc[key, "Lease Effective Date"]
    _keys = [(key[0],i) for i in _leases[
        _leases["Lease Effective Date"] < eff_date].index if
             np.any(bid_data.index.isin([(key[0],i)]))]

    
    if len(_keys):
        # if it's bid on it's own or been the largest player
        nn.loc[key, "Previous Interest"] =  int(company_name in
                                    bid_data.loc[_keys,"Company Name"])
        
        # if it's bid in a consortia
        if not nn.loc[key, "Previous Interest"]:
            indx = consortia.index.isin(_keys)
            nn.loc[key, "Previous Interest"] =  int(company_name in
                            consortia.loc[indx, "Company Name"].values)
        
        
    nn.loc[key, "Min Bid"] = bid_data.loc[key, "BID"].min()
    nn.loc[key, "Max Bid"] = bid_data.loc[key, "BID"].max()  
    nn.loc[key, "MROV"] = bid_data.loc[key, "MROV"].max()            
    nn.loc[key, "MROV"] = bid_data.loc[key, "MROV"].max()            
    
    nn.loc[key,"Oil Production"] = oil_prod
    nn.loc[key,"Gas Production"] = gas_prod
    nn.loc[key,"Max Water Depth"] = water_depth
    nn.loc[key,"Producing Completions"] = num_compl

def load_dash_data(test_area, base_directory, freeze_data):
    owners, _, = boem_lease_owner(base_directory)

    leases = boem_leases(base_directory)
    leases.sort_index(inplace=True)
    leases["Lease Effective Date"] = \
                    leases["Lease Effective Date"].dt.to_period(period_size)
    leases["Lease Expiration Date"] = \
                    leases["Lease Expiration Date"].dt.to_period(period_size)


    qdata = pd.read_csv(join(base_directory, "Quarterly_ProdData.csv"))
    qdata.set_index(["Lease Number", "Production Date"], inplace=True)

    planning_areas = pd.read_csv(join(base_directory,
                                      "blocks_by_planningarea.csv"))
    
    planning_areas.set_index("AREABLK", inplace=True)

    bid_data, winning_bids, consortia = freeze_bids(freeze_data)
    consortia.set_index(["AREABLK", "Lease Number"], inplace=True)

    nn = read_curated_neighbourhoods(base_directory)
    nn["Lease Effective Date"] = \
                    nn["Lease Effective Date"].dt.to_period(period_size)
    nn["Lease Expiration Date"] = \
                    nn["Lease Expiration Date"].dt.to_period(period_size)

    return leases, owners, qdata, bid_data, winning_bids, consortia, nn




def slice_by_area(test_area, leases, owners, qdata,
                  bid_data, winning_bids, consortia, nn):
    # --------- #
    header = provide_current(test_area, leases, owners, qdata, cumulative=True)
    init_block_data(header)
    combine_block_data(header, header.index[0], leases,
                       qdata, bid_data, winning_bids, consortia)


    # this block
    block_past_leases = past_leases(test_area, leases,
                                    owners, qdata, cumulative=True)
    for key, val in block_past_leases.iterrows():
        if val["Company Name"] == "None":
            continue
        if not np.any(winning_bids.index.isin([(test_area, key)])):
            continue
        
        block_past_leases.loc[key,"Company Name"] = \
                winning_bids.loc[(test_area, key),"Company Name"].values
        
    init_block_data(block_past_leases)
    for key, _ in block_past_leases.iterrows():
        combine_block_data(block_past_leases, key, leases,
                             qdata, bid_data, winning_bids, consortia)

    nn = nn.loc[test_area].copy()
    init_block_data(nn)
    for key, _ in nn.iterrows():
        combine_block_data(nn, key, leases, qdata,
                             bid_data, winning_bids, consortia)


    header.reset_index(inplace=True)    
    block_past_leases.reset_index(inplace=True)        
    nn.reset_index(inplace=True)        
        
    nn.rename(columns={"Lease Status Code" : "Lease Code",
                       "AREABLK_NN": "AREABLK"},
              inplace=True)


    for key,val in header.groupby("AREABLK"):
        header.loc[val.index[1:], "AREABLK"] = ""

    for key,val in block_past_leases.groupby("AREABLK"):
        block_past_leases.loc[val.index[1:], "AREABLK"] = ""

    for key,val in nn.groupby("AREABLK"):
        nn.loc[val.index[1:], "AREABLK"] = ""

    nn.dropna(subset=["Lease Effective Date"], inplace=True)
    nn = nn[nn['Company Name'] != "Unknown"]

    
    return header, block_past_leases, nn

def convert_datetimes(df):
    for key,val in df.dtypes.iteritems():
        if isinstance(val, pd.core.dtypes.dtypes.PeriodDtype):
            df[key] = df[key].astype(str)
    


def slice_well_by_area(test_area, leases, nn):
    # --------- #
    header = leases.loc[[test_area]].copy()

    blks = nn.loc[test_area].index.get_level_values(0)    
    footer = leases.loc[blks].copy()

    header.reset_index(inplace=True)    
    footer.reset_index(inplace=True)        

    for key,val in header.groupby("AREABLK"):
        header.loc[val.index[1:].values, "AREABLK"] = ""

    for key,val in footer.groupby("AREABLK"):
        footer.loc[val.index[1:].values, "AREABLK"] = ""
            
    header.drop(columns=["Area Code", "Block Number",
                         "Block Max Water Depth (meters)"], inplace=True)
    footer.drop(columns=["Area Code", "Block Number",
                         "Block Max Water Depth (meters)"], inplace=True)
        
    return header, footer
