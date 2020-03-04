from os.path import dirname, join
from elvis.visualization.utils import load_dash_data, slice_by_area
from elvis import datasets
from elvis.io.boem_from_file import (boem_leases,
                                     boem_lease_by_owner,
                                     get_neighbourhood_leases, 
                                     get_blocks, 
                                     read_curated_neighbourhoods,
                                     freeze_bids,
                                     load_num_wells,
                                     boem_platform_structures)

from elvis.visualization.mapping import (bathymetry_underlay,
                                         create_map_from_geojson,
                                         colors as company_colors,
                                         geojson_underlay)

from elvis.model.backtesting import (BidData, KernalAvgByBlock,
                                     LeaseBasedFeatures, explanatory_vars)

import click
from itertools import chain
import geopandas as gpd
import numpy as np
import pandas as pd
import tqdm

@click.command()
@click.option('--mini', default=-1, help='min i')
@click.option('--maxi', default=-1, help='max i')
def run(mini, maxi):
    base_directory = dirname(datasets.__file__)
    freeze_data = join(base_directory, 'Freeze_Data\ 12_4_2019')

    # figure out a "20m arc length but in radians"
    EARTH_RADIUS_M = 1000*6378.1
    TEN_MILES = 10 * 1.6 *1000
    TWENTY_MILES = 20 * 1.6 *1000

    #
    period_size = 'Q'
    # date range for backtesting
    date_range = pd.date_range(start=pd.datetime(2003,1,1), 
                               end=pd.datetime(2020,1,1), 
                               freq=period_size)
    periods = date_range.to_period(period_size)
    current_period = pd.Timestamp.now().to_period('Q')

    # ------- #

    blocks = get_blocks(base_directory)
    blocks.set_index("AREABLK", inplace=True)

    leases = boem_leases(base_directory)
    leases.sort_index(inplace=True)
    leases["Lease Effective Date"] = \
                        leases["Lease Effective Date"].dt.to_period(period_size)
    leases["Lease Expiration Date"] = \
                        leases["Lease Expiration Date"].dt.to_period(period_size)
    # drop bad data
    leases.dropna(subset=["Lease Effective Date"], inplace=True)

    wells = load_num_wells(base_directory)

    qdata = pd.read_csv(join(base_directory, "Quarterly_ProdData.csv"))
    qdata['Production Date'] = [pd.Period(i) for i in qdata['Production Date'].values]
    qdata.set_index(["Lease Number", "Production Date"], inplace=True)

    planning_areas = pd.read_csv(join(base_directory,"blocks_by_planningarea.csv"))
    planning_areas.set_index("AREABLK", inplace=True)

    bid_data, winning_bids, consortia = freeze_bids(freeze_data)
    consortia.set_index(["AREABLK", "Lease Number"], inplace=True)

    # only consider leases that have associated bids (edge cases)
    _lease = winning_bids.merge(leases, how="inner",
                                on=["AREABLK", "Lease Number"], right_index=True)
    leases = _lease[leases.columns]

    # infrastructure
    platform_structures = boem_platform_structures(base_directory)
    platform_structures["AREABLK"] = platform_structures["AREABLK"].str.replace("\W","")
    platform_structures.dropna(subset=["AREABLK"], inplace=True)
    platform_structures.set_index("AREABLK", inplace=True)
    #
    platform_structures["Install Period"] = \
                    platform_structures["Install Date"].dt.to_period(period_size)
    platform_structures["Removal Period"] = \
                    platform_structures["Removal Date"].dt.to_period(period_size)

    # object for dealing with averaging over neighbourhood
    kde = KernalAvgByBlock(base_directory)

    # ------- #
    blocks = get_blocks(base_directory)
    blocks.set_index("AREABLK", inplace=True)

    bid_data, winning_bids, consortia = freeze_bids(freeze_data)
    bd = BidData(bid_data=bid_data, 
             start_period=pd.Period(pd.datetime(2009,1,1), "Q"))
    # ------- #
    all_blocks = blocks.index.values
    # create generator of explanatory variables.
    lbfs = {area_block:LeaseBasedFeatures(area_block=area_block, kde=kde) for 
            area_block in tqdm.tqdm(all_blocks)}
    # ------- #
    auctions = bd.hold_auctions(leases)
    # ------- #
    for i, val in enumerate(bd.hold_auctions(leases)):
        
        if i < mini or i >= maxi:
            continue

        period, _leases, mlot = val

        area_blocks = mlot.index.get_level_values(0).values
        # everything in a 20-mile radius of every block bid on
        aoi = list(chain(*[list(lbfs[blk]._nn.keys()) for blk in area_blocks]))
        aoi = np.unique(aoi)

        result = [explanatory_vars(lbfs[arr], _leases, bid_data,
                                   wells, qdata, platform_structures, period) for 
                  arr in tqdm.tqdm(aoi)]
        #
        X = np.vstack(result)

        has_bid = [i in mlot.index.get_level_values(0) for i in aoi]
        
        bid = np.zeros(len(has_bid))
        bid[:] = -1
        bid[has_bid] = [mlot.loc[i,"BID"].values[0] for i in aoi if i in 
                        mlot.index.get_level_values(0)]
        has_bid = np.int_(has_bid)
        
        # if we couldn't figure it out:
        indx = ~np.any(np.isnan(X), axis=1)
        X = X[indx,:]
        bid = bid[indx]
        has_bid = has_bid[indx]
        
        # use the "i" index for the lease auction to reloade over time.
        np.save(join(base_directory, "bid-model-1", "X-{}.npy".format(i)),X)
        np.save(join(base_directory, "bid-model-1", "bid-{}.npy".format(i)),bid)
        np.save(join(base_directory, "bid-model-1", "has_bid-{}.npy".format(i)),has_bid)    
        mlot.to_csv(join(base_directory, "bid-model-1", "bids-{}.csv".format(i)))    
    
if __name__ == "__main__":
    run()

