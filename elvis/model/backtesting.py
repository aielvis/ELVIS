from os.path import dirname, join

import numpy as np
import pandas as pd

from sklearn.neighbors import BallTree
from scipy.stats import norm
from traits.api import (Any, Array, Bool, Dict, HasStrictTraits,
                        Instance, List, Unicode)

# figure out a "20m radius but in radians"
EARTH_RADIUS_M = 1000*6378.1
TEN_MILES = 10 * 1.6 *1000
TWENTY_MILES = 20 * 1.6 *1000

from elvis.io.boem_from_file import get_blocks

class KernalAvgByBlock(HasStrictTraits):    
    area_blocks = Array
    btree = Any
    centers = Array
    
    def __init__(self, base_directory):
        # for every block, get its neighbourhood and 
        blocks = get_blocks(base_directory, planning_region=False)
        centers = np.vstack([np.deg2rad(i.coords[0]) for i in
                             blocks['geometry'].centroid])
        centers = centers[:,::-1]
        
        # save this
        self.centers = centers
        self.area_blocks = blocks["AREABLK"].values
        # I need both the locations and the weights
        self.btree = BallTree(centers, metric="haversine")

    def kernel_weighted_avg(self, area_block,
                            radius=TWENTY_MILES/EARTH_RADIUS_M):
        """ The radius will be interpreted as 2*\sigma in the weights,
            i.e. we gather all the data we need within 2*\sigma. 
        """
        
        indx = self.area_blocks == area_block
        if not np.any(indx):
            raise RuntimeError(
                "Area block {} not in blocks.".format(area_block))
        
        result = self.btree.query_radius(self.centers[indx,:], 
                                         r=radius, return_distance=True)
        target_blocks, r = result
        weights = norm(loc=0, scale=(0.5*radius)).pdf(r[0])
        weights /= np.sum(weights)
        
        _area_blocks = self.area_blocks[target_blocks[0]]
        return {key:val for key,val in zip(_area_blocks,weights)}


class LeaseBasedFeatures(HasStrictTraits):
    """ Generate Features associted with lease ownership, by neighbourhood
        and roll by period."""
    area_block = Unicode
    
    # holds area_block and kernel weight
    _nn = Dict
    
    def __init__(self, kde, **kwargs):
        # construct traits
        super().__init__(**kwargs)
        
        self._nn = kde.kernel_weighted_avg(self.area_block, 
                                      radius=0.66*TEN_MILES/EARTH_RADIUS_M)
        
    def clip_lease_back_contract(self, leases, period):
        """ Get the previous contract if on exists"""
        _leases = leases.copy()            
        _leases.sort_values("Lease Effective Date", inplace=True)
        # lease started before the current period
        _mask_start = _leases["Lease Effective Date"] < period
        # ends before the current period
        _mask_end = _leases["Lease Expiration Date"] < period

        _prev_leases = _leases[np.logical_and(_mask_start, _mask_end)]

        return (period, 
               _leases.loc[[val.index[-1] for 
                                key,val in _prev_leases.groupby(level=0)]])
            
    def clip_lease_front_contract(self, leases, period):
        """This thing generates lease dataframes roll by 'current contract' or 
           last expired contract."""
        _leases = leases.copy()
        # the lease started prior to the current quarter (or before)
        _mask_start = _leases["Lease Effective Date"] < period
        # the lease ends after or during current quarter          
        _mask_end = np.logical_or(_leases["Lease Expiration Date"] >= period, 
                                  _leases["Lease Expiration Date"].isna())

        _leases = _leases[np.logical_and(_mask_start, _mask_end)]
        _leases.loc[_leases["Lease Expiration Date"] > period,
                    "Lease Expiration Date"] = pd.NaT            
        return (period, _leases)            
    
    def _clip_lease_back_contract(self, leases, period):
        """The previous contract, or if expired and not renewed, the most
           recent.
        """
        _leases = leases.copy()
        # lease started before the current period
        _mask_start = _leases["Lease Effective Date"] < period
        # ends before the current period
        _mask_end = _leases["Lease Expiration Date"] < period
        _prev_leases = _leases[np.logical_and(_mask_start, _mask_end)]

        # the lease started prior to the current quarter (or before)
        _mask_start = _leases["Lease Effective Date"] < period
        # the lease ends after or during current quarter          
        _mask_end = np.logical_or(_leases["Lease Expiration Date"] >= period, 
                                  _leases["Lease Expiration Date"].isna())    
        _current_leases = _leases[np.logical_and(_mask_start, _mask_end)] 

        _rel_blocks = set(_prev_leases.index.get_level_values(0)).difference(
                            set(_current_leases.index.get_level_values(0)))

        return (period,_leases.loc[_rel_blocks].copy())
    
    def clip_lease_by_past(self, leases, period):
        """This thing generates lease dataframes roll by past contracts."""
        _leases = leases.copy()
        _mask_end = _leases["Lease Expiration Date"] < period            
        _leases = _leases[_mask_end]
        return (period,_leases)
        
    def clip_lease_by_period(self, leases, period):
        """This thing generates lease dataframes roll by 'current contract' or 
           last expired contract."""
        _leases = leases.copy()
        # the lease started prior to the current quarter (or before)
        _mask_start = _leases["Lease Effective Date"] < period
        # the lease ends after or during current quarter          
        _mask_end = np.logical_or(_leases["Lease Expiration Date"] >= period, 
                                  _leases["Lease Expiration Date"].isna())

        _leases = _leases[np.logical_and(_mask_start, _mask_end)]
        _leases.loc[_leases["Lease Expiration Date"] > period,
                    "Lease Expiration Date"] = pd.NaT   
        
        return (period,_leases)

    def clip_lease_data_by_period(self, leases, period):
        """This thing generates lease dataframes roll by 'current contract' or 
           last expired contract."""
        _leases = leases.copy()
        # the lease started prior to the current quarter (or before)
        _mask_start = _leases["Lease Effective Date"] < period
        # the lease ends after or during current quarter          
        _mask_end = np.logical_or(_leases["Lease Expiration Date"] >= period, 
                                  _leases["Lease Expiration Date"].isna())

        _leases = _leases[np.logical_and(_mask_start, _mask_end)]
        _leases.loc[_leases["Lease Expiration Date"] > period,
                    "Lease Expiration Date"] = pd.NaT            
        return (period,_leases)
            
    def rolling_relinquished_well(self, leases, wells, period, code="RELINQ"):
        _leases = leases.loc[list(self._nn.keys())].copy()

        period, back = self.clip_lease_back_contract(_leases, period)
        _, front = self.clip_lease_front_contract(_leases, period)

        # filter by lease status
        back = back[back["Lease Status Code"] == code]

        # find all lease blocks with a back but not current contract
        blks = set(back.index.get_level_values(0)).difference(
                                    front.index.get_level_values(0))

        # filter by relinquished with no front contract
        back = back.loc[list(blks)]

        # relinquished back contracts with explortation wells.
        # don't have to worry about Spud date because the lease 
        # is already been relinquished, it's in the past.
        back = back.join(wells, how="inner")
        return (np.sum([self._nn[key] for key,val in back.groupby(level=0)]))
            
    def rolling_relinquished(self, leases, period, code="RELINQ"):
        _leases = leases.loc[list(self._nn.keys())].copy()
        
        period, back = self.clip_lease_back_contract(_leases, period)
        _, front = self.clip_lease_front_contract(_leases, period)

        # 
        back = back[back["Lease Status Code"] == code]

        # find all lease blocks with a back but not current contract
        blks = set(back.index.get_level_values(0)).difference(
                            front.index.get_level_values(0))
        # block weights 
        return(np.sum([self._nn[blk] for blk in blks]))

    def open_blocks(self, leases, period):
        """
        A block is open if it doesn't have a lease associated
        """
        _leases = leases.loc[list(self._nn.keys())]
        period, lease = self.clip_lease_by_period(_leases, period)
        return (np.sum([i[1] for i in self._nn.items() if i[0] 
                                not in lease.index.get_level_values(0)]))

    def past_leases(self, leases, period):
        """ Weighted number of past leases in the neighbourhood."""     
        _leases = leases.loc[list(self._nn.keys())].copy()
        period, lease = self.clip_lease_by_past(_leases, period)
            # sum weight for every past contract.            
        return (np.sum([self._nn[key[0]] for key,val in lease.iterrows()]))

    def current_leases(self, leases, period):
        """ Weighted number of current in the neighbourhood."""     
        _leases = leases.loc[list(self._nn.keys())].copy()
        period, lease = self.clip_lease_by_period(_leases, period)
            # sum weight for every past contract.       
        return (np.sum([self._nn[key[0]] for key,val in lease.iterrows()]))

        
    def quarterly_production(self, leases, qdata, period, value="oil"):        
        """In lieu of field EUR, use the quarterly production.
           Nugget is the value to use for the current block
        """
        
        values = {"oil" : "Lease Oil Production (BBL)",
                  "condensate" : "Lease Condensate Production (BBL)",
                  "gas_well_gas" : "Lease Gas-Well-Gas Production (MCF)",
                  "oil_well_gas" : "Lease Oil-Well-Gas Production (MCF)"}
        _value = values[value]
        

        # inclusive of the current block
        _leases = leases.loc[list(self._nn.keys())]        
        _leases = _leases.join(qdata, how="inner")
        _leases = _leases.swaplevel(0,1)
        _leases.index.names = ["AREABLK", "Lease Number", "Period"]
            
        _nn = self._nn.copy()       
        period,lease = self.clip_lease_by_period(_leases, period)
        lease = lease.loc[lease.index.get_level_values('Period') == period]
        
        return (np.sum([_nn[key[0]] * val[_value] for
                        key,val in lease.iterrows()]))
            
            
    def platform_structures(self, platform_structures, period):
        """Use any and all structures"""            
        _nn = self._nn.copy()

        def getval(area_block, period):
            
            install_period = platform_structures.loc[[area_block],
                                                     "Install Period"]
            removal_period = platform_structures.loc[[area_block],
                                                     "Removal Period"]         
            removal_period[pd.isna(removal_period)] = period + 1               
                
            return np.sum(np.logical_and(install_period <= period,
                                         removal_period > period))
    
        return (np.sum([getval(blk, period)*_nn[blk] for blk in _nn.keys() if \
                                blk in platform_structures.index ]))
    
    def wells(self, wells, period):
        """Use any and all structures"""            
        _nn = self._nn.copy()

        def getval(area_block, period):
            # count spud well in next quarter
            return np.sum(
                wells.loc[area_block].index.get_level_values(1) < period)

        # FIX ME we need to normalize the _nn
        return (np.sum([getval(blk, period)*_nn[blk] for
                        blk in _nn.keys() if \
                                blk in wells.index ]))


    def water_depth(self, leases):
        """ The lease data contains the "water depth", but for the first time a
            block is leased, this information maybe unknown.
        """
        
        
        has_data = [blk for blk in self._nn.keys() if blk in leases.index]        
        if len(has_data) < 1:
            # unknow depth, we loose this data point.
            return np.nan

        # water depth needs to be properly normalized; it's not weighted
        # in the sense of a price, we really just want the average depth in a region.
        norm = np.sum([val for blk,val in self._nn.items() if blk in leases.index])
        
        depth = np.sum([leases.loc[blk, "Block Max Water Depth (meters)"].max()*val
                           for blk,val in self._nn.items() if 
                               blk in leases.index])/norm
        
        return depth
    
    def average_bid(self, leases, bid_data, period):
        # this is leathal if there's information flow, so
        # back-date the period
        _leases = leases.loc[list(self._nn.keys())]        
        _, front = self.clip_lease_front_contract(_leases, period-1)
        
        if len(front) > 0:
            _bid_data = bid_data.loc[front.index]
            wts = np.sum([self._nn[key[0]] for key,val in _bid_data.iterrows()])
            bavg = np.sum([val["BID"] * self._nn[key[0]] for 
                           key,val in _bid_data.iterrows()])/wts
            return bavg
        else:
            return 0.
        
    
    def high_bid(self, leases, bid_data, period):
        _leases = leases.loc[list(self._nn.keys())]        
        _, front = self.clip_lease_front_contract(_leases, period-1)
        if len(front) > 0:
            _bid_data = bid_data.loc[front.index]
            return np.max([val["BID"] for key,val in _bid_data.iterrows()])
        else:
            return 0.
        
class BidData(HasStrictTraits):
    """
    """
    bid_data = Instance(pd.DataFrame)
    start_period = Instance(pd.Period)
    auctions = Array
        
    #If you want to back-date information so you can't see anything 
    #up to the previous quarter.
    backdate = Bool(False)

    def __init__(self, **kwargs):
        #        
        super().__init__(**kwargs)

        #
        self.auctions = np.unique(self.bid_data["SALEDATE"])
        if "start_period" not in kwargs.keys():
            self.start_period = pd.Period(pd.datetime(2003,1,1), "Q")
    
    def hold_auctions(self, leases, period_size='Q'):
        """
        To train a model we can only use the data we have.
        
        The rule is that you get all the lease bases features you 
        want, up to the previous quarter. That should simulate a flow         
        """
        for auction in self.auctions:
            auction_period = pd.Period(auction, "Q")
            
            if auction_period < self.start_period:
                continue
            #
            _bid_data = self.bid_data[self.bid_data["SALEDATE"] == auction]
            
            _blocks = _bid_data.index.get_level_values(0)
            _lease_numbers_bid = _bid_data.index.get_level_values(1)
            
            two_bids = [i[0] for i in
                        _bid_data[_bid_data["BIDORDER"] == 2].index]
            first_bids = list(set(_blocks).difference(set(two_bids)))
            
            # mlot!
            df = _bid_data.loc[two_bids,["BIDORDER", "BID", "MROV"]].copy()
            _val_1 = df[df["BIDORDER"] == 1]
            _val_2 = df[df["BIDORDER"] == 2]

            mlot = {key:_val_1.loc[key,"BID"] - np.max(val["BID"] - 
                                        val["MROV"]) for
                    key,val in _val_2.iterrows()}
            mlot_2 = pd.DataFrame(columns=["MLOT"], 
                                         data=mlot.values(), 
                                             index=mlot.keys())     
            mlot_2["BID"] = _val_2["BID"]
            mlot_2["MROV"] = _val_2["MROV"]

            mlot_2.index.names = ["AREABLK", "Lease Number"]

            mlot_1 = _bid_data.loc[first_bids,["BIDORDER", "BID", "MROV"]]
            mlot_1["MLOT"] = mlot_1["BID"] - mlot_1["MROV"]
            mlot_1.drop(columns=["BIDORDER"], inplace=True)

            mlot = pd.concat([mlot_1, mlot_2], sort=True)

            #
            indx = set(mlot.index).intersection(set(leases.index))
            
            # I need a lease for every bid
            _leases = leases.copy()
            _leases = _leases[_leases["Lease Effective Date"] < auction_period]
            _leases.sort_index(inplace=True)
            
            #
            mlot = mlot.loc[list(indx)]
            mlot.sort_index(inplace=True)
            
            yield auction_period, _leases, mlot

def explanatory_vars(lbf, leases, bid_data,
                     wells, qdata, platform_structures, period):
    """ Generates a rolling set of data with these explanatory vars.
        Suggest saving the result to disk and reloading for training/prediction.
    """
    return (lbf.rolling_relinquished_well(leases, wells, period),
            lbf.rolling_relinquished(leases, period),
            lbf.open_blocks(leases, period),
            lbf.past_leases(leases, period),
            lbf.current_leases(leases, period),
            lbf.quarterly_production(leases, qdata, period,
                                     value="oil"),
            lbf.quarterly_production(leases, qdata, period,
                                     value="condensate"),
            lbf.quarterly_production(leases, qdata, period,
                                     value="gas_well_gas"),
            lbf.quarterly_production(leases, qdata, period,
                                     value="oil_well_gas"),
            lbf.platform_structures(platform_structures, period),
            lbf.wells(wells, period),
            lbf.average_bid(leases, bid_data, period),
            lbf.high_bid(leases, bid_data, period),
            lbf.water_depth(leases),)
            
