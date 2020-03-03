""" Convenience functions for loading BOEM data.
"""

from os.path import join
import json
import re

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity, KDTree
import shapefile

structure_type = {
    "CAIS" : "Caisson", 
     "CT" : "Compliant tower",  
     "FIXED" : "Fixed Leg Platform",
     "FPSO" : "Floating production, storage, and offloading",
     "MOPU" : "Mobile Production Unit",
     "MTLP" : "Mini Tension Leg Platform",
     "SEMI" : \
        "Semi Submersible (Column Stabilized Unit) Floating \
        Production System",
     "SPAR" : "SPAR Platform - floating production system",
     "SSANC" : \
          "Fixed anchors or mooring piles used to secure a \
                structure to the seafloor.",
     "SSMNF" : "Subsea Manifold",
     "SSTMP" : "Subsea templates",
     "TLP" : "Tension leg platform",
     "UCOMP" : "Underwater completion or subsea caisson",
     "WP" : "Well Protector"}

platform_type = {
    "CAIS" : "Caisson", 
     "CT" : "Compliant tower",  
     "FIXED" : "Fixed Leg Platform",
     "FPSO" : "Floating production, storage, and offloading",
     "MOPU" : "Mobile Production Unit",
     "MTLP" : "Mini Tension Leg Platform",
     "SEMI" : \
        "Semi Submersible (Column Stabilized Unit) Floating \
        Production System",
     "SPAR" : "SPAR Platform - floating production system",
     "SSANC" : \
          "Fixed anchors or mooring piles used to secure a \
                structure to the seafloor.",
     "SSMNF" : "Subsea Manifold",
     "SSTMP" : "Subsea templates",
     "TLP" : "Tension leg platform",
     "UCOMP" : "Underwater completion or subsea caisson",
     "WP" : "Well Protector"}


lease_codes = { 
"CANCEL","Cancelled by the authorized officer.",
"CONSOL","A terminated lease whose acreage has been merged into another lease.",
"DSO","Operations/activities on all or part of lease suspended/temp prohibited on Reg Sup initiative. Lease term extended.",
"EXPIR","A lease whose initial term has ended by operation of law."
"EXTSEG","A lease segregated prior to 1979; held by production from or activity on the original lease."
"NO-EXE","An awarded lease not executed by an authorized official of the Bureau. A lease with the new NO-EXE status is a bid evaluated by RE and deemed acceptable. Adjudication awarded the lease to the company. The company chose to accept the lease and paid the balance of the bonus. However, the agency cannot sign the lease and make it effective/active (there could be various reasons for this decision). Therefore the bid never became a lease.",
"NO-ISS","An awarded lease not executed by the bidder(s).",
"OPERNS","Initial term extended because of activity on the leased area.",
"PR DSO","Initial term extended by order of the director.",
"PR SOO","Initial term granted at request of the operator.",
"PRIMRY","A lease within the initial term of the contract (5, 8, or 10 years).",
"PROD","A lease held by production of a mineral.",
"REJECT","A high bid rejected by the authorized officer.",
"RELINQ","A lease voluntarily surrendered by the record title holders.",
"RENGEN","Renewable Energy Generating Term",
"RENOPR","Renewable Energy Operations Term",
"RENPT","Renewable Energy Preliminary Term",
"RENPTPS","Renewable Energy Preliminary Term Pending SAP Review",
"RENSAT","Renewable Energy Site Assessment Term",
"RENSPC","Renewable Energy Site Assessment Pending COP Review",
"RENSUS","Renewable Energy Suspensions",
"SOO","Initial term extended due to ordering or appro by Dir of SOO",
"SOP","Initial term extended due to ordering or appro by Dir of SOP",
"TERMIN","A lease extended beyond its primary terms and has ended by operation of law.",
"UNIT","A lease (or portion thereof) included in an approved unit agreement."}

def contours_to_geojson(base_directory, file_prefix,
                        feature=None, correct_bbox=True):
    """ provide feature as a key/val; the key should be contained 
        in the "geo_json" feauters : properties """
    
    gom_shape = join(base_directory, file_prefix + ".shp")
    gom_dbf = join(base_directory, file_prefix + ".dbf")
    gom_shx = join(base_directory, file_prefix + ".shx")

    reader = shapefile.Reader(gom_shape)
    shp = reader.load_dbf(gom_dbf)
    shx = reader.load_shx(gom_shx)

    contour_json = reader.__geo_interface__

    if "bbox" in contour_json:
        # corrects json issue    
        contour_json['bbox'] = tuple(contour_json['bbox'])

    if feature is not None:
        key, val = feature    
        contour_json['features'] = [i for i in contour_json['features'] if 
            i['properties'][key].lower() == val]
        
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
    boem_lease_owner["Company Name"] = \
                boem_lease_owner["Company Name"].str.replace('\W','')
    
    # some leases are owned by a consortia, use the largest partner
    boem_lease_owner.sort_values(["Lease Number", "Assignment Pct"], 
                                 inplace=True)
    # label where 
    boem_lease_owner["Consortia"] = boem_lease_owner.duplicated(
        subset=["Lease Number"]
    )
    
    # keep a record of the consortia 
    consortia_leases = boem_lease_owner[
        boem_lease_owner.duplicated(subset=["Lease Number"], keep=False)].copy()
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

    boem_leases.sort_values("Lease Number", inplace=True)
    
    # block numbers aren't ints 
    boem_leases["Block Number"] = \
                        boem_leases["Block Number"].str.replace(' +', '')

    # AREABLK same as AC_BLK in shape files
    # FIXME - check codes like AT001 <-> AT1
    boem_leases["AREABLK"] = boem_leases["Area Code"] + \
                             boem_leases["Block Number"]

    boem_leases.set_index(["AREABLK", "Lease Number"], inplace=True)
    
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

def freeze_bids(freeze_data):
    bid_dates = ["SALEDATE"]
    def bid_date_parser(inp):
        val = inp.split('.')[0]
        return pd.datetime(int(val[:4]), 
                           int(val[4:6]), 
                           int(val[6:]))

    # A "bid" that is duplicate, lease number and bidorder
    # represents a consortia bid
    bid_types = {"BIDORDER" : int,
                 "BID" : float,
                 "PCTSHARE" : float}

    bid_data = pd.read_csv(join(freeze_data, "Bidst.txt"),
                           parse_dates=bid_dates, 
                           date_parser=bid_date_parser,
                           dtype=bid_types)

    #rename some stuff
    bid_data.rename(columns={"SORTNAME" : "COMPANY NAME"}, inplace=True)
    bid_data.rename(columns={"LEASENUM" : "Lease Number"}, inplace=True)
    bid_data.rename(columns={"COMPANY NAME" : "Company Name"}, inplace=True)

    #
    bid_data["Company Name"] = bid_data["Company Name"].str.lower()
    bid_data["Company Name"] = bid_data["Company Name"].str.replace('\W','')
    
    # normalize things like "WC010" -> "WV10"
    def split_on_zero(val):
        res = re.match("([A-Z]+)(0{1,})([1-9]{1,}[0-9]{0,})", val)
        if res:
            grps = res.groups()
            return grps[0] + grps[-1]
        else:
            return val
        
    bid_data["AREABLK"] = [split_on_zero(i) for i in bid_data["AREABLK"]]

    # some data has white space for the lease number
    indx = bid_data["Lease Number"].str.match("( +)")

    bad_lease_number = bid_data[indx].copy()
    bid_data.drop(bid_data.index[indx], inplace=True)

    bid_data.sort_values(["Lease Number", "BIDORDER", "PCTSHARE"], inplace=True)

    bid_data["Consortia"] = bid_data.duplicated(
        subset=["Lease Number", "BIDORDER"], keep=False)
    consortia_bids = bid_data[bid_data["Consortia"]].copy()

    bid_data.drop_duplicates(subset=["Lease Number", "BIDORDER"],
                             keep="last", inplace=True)

    winning_bids = bid_data[bid_data["BIDORDER"] == 1]

    # unique multi-index lease no/bidorder
    bid_data.set_index(["AREABLK", "Lease Number"], inplace=True)
    winning_bids.set_index(["AREABLK", "Lease Number"], inplace=True)
    
    return bid_data, winning_bids, consortia_bids


def winning_bid_by_block(base_directory, freeze_data):
    """ Associate geospatial information with the bids.
    base_directory -  place where the boem data is stored.
    freeze_directory - location of freeze data, could be dynamic.
    """

    lease_owners_by_block = get_blocks_by_owner(base_directory)    
    _, winning_bids, _ = freeze_bids(freeze_data)

    
    bid_by_block = lease_owners_by_block.join(winning_bids, 
                                              how="inner", 
                                              on=["AREABLK", "Lease Number"],
                                              rsuffix=" bid", 
                                              lsuffix=" owner")

    bid_by_block.rename(columns={
        'Company Name bid' : 'Original Bidder', 
        'Company Name owner' : 'Company Name',
        'Consortia bid' : 'Bid by Consortia',
        'Consortia owner' : 'Consortia',
    }, inplace=True)
    
    return bid_by_block

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

def get_blocks(base_directory, planning_region=False):
    block_directory = join(base_directory, "blocks")
    block_shape = join(block_directory, "blocks.shp")

    block_data = gpd.read_file(block_shape)
    block_data.rename(columns={"AC_LAB" : "AREABLK"}, inplace=True)

    if planning_region:
        # add meta information about planning area
        planning_area = gpd.read_file(join(base_directory, 'Gulf_PlanningAreas'))    

        indx0 = [val['geometry'].centroid.within(planning_area['geometry'][0])
                 for key,val in block_data.iterrows()]
        indx1 = [val['geometry'].centroid.within(planning_area['geometry'][1])
                 for key,val in block_data.iterrows()]
        indx2 = [val['geometry'].centroid.within(planning_area['geometry'][2])
                 for key,val in block_data.iterrows()]

        block_data["PlanningArea"] = "NaN"
        block_data.loc[indx0, "PlanningArea"] = planning_area['TEXT_LABEL'][0]
        block_data.loc[indx1, "PlanningArea"] = planning_area['TEXT_LABEL'][1]
        block_data.loc[indx2, "PlanningArea"] = planning_area['TEXT_LABEL'][2]
    
    return block_data

def get_blocks_by_owner(base_directory, owner=None):
    """ Supply an owner e.g. "equinor" if you want to have a single owner, 
        otherwise the data will contain all owned leases."""
    
    owned_leases, _ = boem_lease_by_owner(base_directory, owner=owner)
    blocks = get_blocks(base_directory)

    # keeps it through the merge
    owned_leases["Lease Number"] = \
                owned_leases.index.get_level_values('Lease Number').values
    
    owned_leases_by_block = blocks.merge(owned_leases,
                                  how="inner",
                                  left_on="AREABLK",
                                  right_on="AREABLK")
    
    owned_leases_by_block.set_index(["AREABLK", "Lease Number"], inplace=True)

    
    return owned_leases_by_block


def boem_pipeline_meta(base_directory, current=pd.datetime(2000,1,1)):
    """ The data seems to be somewhat incomplete for very old pipelines. 
        Use current leasing information to gauge if a pipeline, without 
        and Install Date is still in planning.
    """

    
    leases = boem_leases(base_directory)
    # Some are approved by not installed, or in fact, may not have been
    # approved, if we can relate them to a current lease we'll suppose
    # they are "planned", unless they have "Abandonment" dates.
    owned_leases, consortia = boem_lease_by_owner(base_directory)
    
    boem_pipeline_dtypes = {"Segment Number" : int}
    boem_pipeline_dates = ["Approved Date", "Install Date",
                           "Out of Service Date", 
                           "Temporary Cessation of Operations Date",
                           "Flush and Fill Date'",
                           "Abandonment Approval Date",
                           "Abandonment Date"]

    pipeline_meta = pd.read_csv(join(base_directory,
                                     "PipelinePermitsSegments.csv"))

    # there's six duplicates, don't seem to have meaningful difference
    pipeline_meta.drop_duplicates(subset="Segment Number",
                                  keep="first", inplace=True)

    pipeline_meta.sort_values("Approved Date", inplace=True)

    # some pipes were approved by then the approval was abandoned and there's 
    # no install date
    never_build = (~pipeline_meta["Abandonment Approval Date"].isna() & 
                   pipeline_meta["Install Date"].isna())
    pipeline_meta.drop(index=pipeline_meta[never_build].index, inplace=True )


    # Try to associate pipeline with current owner of associted lease
    pipeline_meta["Current Lease Owner"] = "NaN"
    has_lease = [i in owned_leases.index for i in
                 pipeline_meta["Originating Lease Number"]]
    lease_owner = [owned_leases.loc[i,"Company Name"] for i in 
                   pipeline_meta["Originating Lease Number"] if
                   str(i) in owned_leases.index]

    #
    current_leases = leases["Lease Effective Date"] > current
    leases = leases[current_leases]

    has_current_lease = [i in leases.index for i in
                         pipeline_meta["Originating Lease Number"]]
    
    indx = (~pipeline_meta["Originating Lease Number"].isna() & 
            pipeline_meta["Install Date"].isna() & 
            has_current_lease)
    pipeline_meta.loc[pipeline_meta[indx].index,
                                     "Install Date"] = pd.datetime.now()
    pipeline_meta["Planned"] = indx

    # otherwise drop everything that's not install or planned
    indx = pipeline_meta["Install Date"].isna()
    pipeline_meta.drop(index=pipeline_meta[indx].index, inplace=True )
    
    # segment number appears to be unique:
    pipeline_meta.set_index("Segment Number", inplace=True)

    return pipeline_meta

def get_pipeline_contours(base_directory):
    
    pipelines_geo = gpd.read_file(join(base_directory,
                                       "ppl_arcs", "ppl_arcs.shp"))
    pipelines_geo.rename(columns={"SEGMENT_NU" : "Segment Number",
                                  "SDE_COMPAN" : "Company Name"},
                         inplace=True)
    
    pipelines_geo["Company Name"] = pipelines_geo["Company Name"].str.lower()
    pipelines_geo["Company Name"] = \
                    pipelines_geo["Company Name"].str.replace('\W','')

    pipelines_geo.set_index("Segment Number", inplace=True)


    return pipelines_geo

def get_pipelines_with_meta(base_directory, by_company=None):


    pipeline_meta = boem_pipeline_meta(base_directory)
    pipelines_geo = get_pipeline_contours(base_directory)
    
    pipelines_geo = pipelines_geo.join(pipeline_meta, how="inner")    

    if by_company is not None:
        #The company originated the pipeline or currently own a lease 
        #asscociated with it.
        indx = ((pipelines_geo["Company Name"].str.contains(by_company) | 
         pipelines_geo["Current Lease Owner"].str.contains(by_company)) &
                ~pipelines_geo["Company Name"].isna())
        
        pipelines_geo = pipelines_geo[indx].copy()

    return pipelines_geo

def boem_platform_structures(base_directory):
    platform_structures = pd.read_csv(join(base_directory, "PlatStruc.csv"))


    platform_structures_dates = ['Install Date', 'Removal Date']
    platform_structures_dtypes = {
                       "Structure Number" : int,
                       "Ptfrm X Location" : float, 
                       "Ptfrm X Location" : float,
                       "Latitude" : float,
                       "Longitude" : float,
                       "Surf N S Dist" : float,
                       "Surf N S Dist" : float, 
                       "Surf E W Dist" : float}
                                              
    platform_structures = pd.read_csv(join(base_directory, "PlatStruc.csv"),
                                      dtype=platform_structures_dtypes,
                                      parse_dates=platform_structures_dates)
                       
    platform_structures["AREABLK"] = platform_structures["Area Code"] + \
                                     platform_structures["Block Number"]
    platform_structures["AREABLK"] = \
                    platform_structures["AREABLK"].str.replace("\W","")

    return platform_structures

def platform_to_geo_json(plat):
    features = []
    for key, val in plat.iterrows():
        _features = json.loads(val.to_json())
        _features["type"] = "Feature"
        _features["properties"] = {"color" : "red",
                                   "marker-size": "small",
                                   "marker-symbol": "bus"}        
        _features["geometry"] = {'type': 'Point',
                                 'coordinates': (val['Longitude'],
                                                 val['Latitude'])}
        
        features.append(_features)    
    return {'type': 'FeatureCollection', 'features' : features}

def platform_by_structure(platform, structure=None):
    """
    Probably not the best way to do this.
    
    Structures descriptions in structure_type.
    """
    if structure not in structure_type and structure is not None:
        raise RuntimeError("Structure {} not contained in should be ")
    
    if structure is not None:
        _structure = platform[platform["Struc Type Code"] == structure].copy()
    else:
        _structure = platform
        
    _structure.__geo_interface__ = platform_to_geo_json(_structure)

    return _structure

def load_regional_prod(base_directory):
    def production_dates(prod):
        prod[prod == "Month Total"] = ""
        prod = pd.to_datetime(prod)
        return prod

    boem_production_dates = ["Production Month/Year"]
    production_by_area = pd.read_csv(join(base_directory, "PBPA.csv"), 
                                     parse_dates=boem_production_dates,
                                     date_parser=production_dates)
    return production_by_area


def _load_production_data(filename):
    def production_dates(prod):
        month = prod["Production Month"].values.astype(int)
        year = prod["Production Year"].values.astype(int)
        return [pd.datetime(i, j, 1) for i,j in zip(year, month)]

    boem_production_by_block_dtypes = {
        "Production Month" : int,
        "Production Year" : int,
        "Lease Oil Production (BBL)" : int,
        "Lease Condensate Production (BBL)" : int,
        "Lease Gas-Well-Gas Production (MCF)" : int,
        "Lease Oil-Well-Gas Production (MCF)" : int,   
        "Lease Water Production (BBL)" : int, 
        "Producing Completions" : int, 
        "Lease Max Water Depth (meters)" : int}

    boem_production_by_block = pd.read_csv(filename, low_memory=False)
    indx = np.logical_or(boem_production_by_block["Production Month"].isna(),
                         boem_production_by_block["Production Month"].isna())
    boem_production_by_block.drop(boem_production_by_block.index[indx],
                                  inplace=True)
    boem_production_by_block["Production Date"] = \
                                production_dates(boem_production_by_block)
    
    for key, val in boem_production_by_block_dtypes.items():
        boem_production_by_block[key] = \
                            boem_production_by_block[key].astype(val)

    boem_production_by_block.set_index("Lease Number", inplace=True)
    return boem_production_by_block

def load_production_data(base_directory):
    lot_1 = _load_production_data(join(base_directory,
                                       "ProdData2007-2010.csv"))
    lot_2 = _load_production_data(join(base_directory,
                                       "ProdData2010-2015.csv"))
    lot_3 = _load_production_data(join(base_directory,
                                       "ProdData2015-2020.csv"))

    return  pd.concat([lot_1, lot_2, lot_3])
    
def load_well_headers(freeze_data):
    """ We're only going to load wells we can associate with AREA BLOCKS. 
        And we'll normalize the naming to be consistent with other data.
    """

    def spud_date_parser(val):
        # some data is "0"
        if int(val) < 19000000:
            # Jan 1900 AD yr/mn!
            if int(val) > 190001:
                return pd.datetime(int(str(val)[:4]),
                                   int(str(val)[4:6]),
                                   1)
            else:
                return pd.NaT
        else:
            return pd.datetime(int(str(val)[:4]),
                               int(str(val)[4:6]), 
                               int(str(val)[6:]))
    
    def split_on_zero(val):
        res = re.match("([A-Z]+)(0{1,})([1-9]{1,}[0-9]{0,})", val)
        if res:
            grps = res.groups()
            return grps[0] + grps[-1]
        else:
            return val

    well_dates = ["SPUDDATE", "WSTATUSDT", "FRSTPRODDT","LASTPRODDT"]
    well_headers = pd.read_csv(join(freeze_data, "Well_Header_GOM3.txt"),
                               date_parser=spud_date_parser,
                               parse_dates=well_dates,
                               low_memory=False)
    
    indx = well_headers["FABBRV"].str.match("[A-Z]{1,}[0-9]{3,4}")
    well_headers = well_headers[indx].copy()

    well_headers["AREABLK"] = [split_on_zero(i) for i in well_headers["FABBRV"]]
    well_headers.set_index(["AREABLK", "WELLNAME"], inplace=True)
    
    return well_headers


def load_discoveries(freeze_data):
    disco_dates = ["ESADATE2", "DISSPUDDT", "DISTDDATE"]
    
    disco = pd.read_csv(join(freeze_data, "Discoveries.txt"),
                        parse_dates=disco_dates)

    leni = np.array([len(val["AREABLOCK"].split(",")) for key,val in
                        disco.iterrows()])
    indx1 = leni < 2
    indxx = leni > 1

    disco_1 = disco[indx1].copy()
    disco_m = disco[indxx].copy()

    unpacked = []
    for key, val in disco_m.iterrows():    
        for area_block in val["AREABLOCK"].split(","):
            _new = val.copy()
            _new["AREABLOCK"] = area_block
            unpacked.append(_new)
    
    result = pd.DataFrame(unpacked)
  
    disco = pd.concat([disco_1, result])
    disco.rename(columns={"AREABLOCK" : "AREABLK", 
                          "LEASENUM" : "Lease Number"}, inplace=True)

    """
    Something strange here:
    """
    macondo = disco[disco["DISC_NICK"] == "MACONDO"].copy()
    disco = disco[disco["DISC_NICK"] != "MACONDO"].copy()

    macondo.set_index(["AREABLK"], inplace=True)
    disco.set_index(["AREABLK"], inplace=True)

    return disco, macondo


def infrastructure_score(blocks, period=pd.Period('1974Q2', 'Q-DEC')):
    blk = blocks.loc[has_infra_by_quarter[period],:]
    blk = blk[~blk.index.isna()].copy()
    
    # centers of every block
    centers = [np.deg2rad(i.centroid.coords.xy) for i in
                     blocks['geometry'].values]
    long = [i[0][0] for i in centers]
    lati = [i[1][0] for i in centers]
    X = np.array((lati, long)).T

    # centers for every block that has well infrastructure at specific time:
    centers = [np.deg2rad(i.centroid.coords.xy) for i in
               blk['geometry'].values]
    long = [i[0][0] for i in centers]
    lati = [i[1][0] for i in centers]
    Xp = np.array((lati, long)).T

    # haversine give "great circle distance"
    kde = KernelDensity(bandwidth=0.001, metric="haversine")
    kde.fit(Xp)
    
    blocks["id"] = blocks.index
    _infra_score = kde.score_samples(X)

    indx = _infra_score > -10
    blks = blocks[indx]
    infra_score = {key:val for key,val in
                        zip(blks["id"].values, _infra_score[indx])}
    
    # return updated blocks dataframe with the score:
    return blks, infra_score


def get_neighbourhood_leases(base_directory, num_nearest=9):
    # for every block, get its neighbourhood and 
    blocks = get_blocks(base_directory, planning_region=False)
    centers = np.vstack([np.deg2rad(i.coords[0]) for i in
                         blocks['geometry'].centroid])
    area_blocks = blocks["AREABLK"]

    # euclidean distance, assume small angle approximation
    kdtree = KDTree(centers)
    # for each entry input its neighbourhood
    neighborhoods = kdtree.query(centers, k=num_nearest)
    # from this I want a dataframe, with every AREABLK as
    # index and then every lease in every neighbouring 
    # AREABLK as an index.
    leases = boem_leases(base_directory)

    leases["AREABLK_leases"] = leases.index.get_level_values(level=0)
    vals = [leases.loc[blocks.iloc[neg]["AREABLK"]].droplevel(level=0)
            for neg in neighborhoods[1]]
    neighbourhood_blocks = pd.concat(vals, keys=blocks["AREABLK"])
    indx = [key[0] == val["AREABLK_leases"] for key,val in
            neighbourhood_blocks.iterrows()]
    neighbourhood_blocks.drop(neighbourhood_blocks[indx].index, inplace=True)
    neighbourhood_blocks.drop(columns=["AREABLK_leases"], inplace=True)

    return neighbourhood_blocks

def get_lease_features(base_directory, leases, well_headers):
    """
    has been leased, relinquished, has been relinquished and has a well drilled.
    """
    lease_area_blocks = leases.index.unique(level=0)
    _has_previous = leases.groupby(level='AREABLK', sort=False).size() > 1
    _has_relinq = (leases["Lease Status Code"] == "RELINQ").groupby(
        level='AREABLK', sort=False).sum() > 0

    has_previous = leases.loc[lease_area_blocks[_has_previous]]
    has_relinq = leases.loc[lease_area_blocks[_has_relinq]]

    # relinq and lease
    relinq_area_blocks = has_relinq.index.unique(level=0)
    _relinq_with_well = [i in well_headers.index for i in relinq_area_blocks]
    relinq_with_well = leases.loc[relinq_area_blocks[_relinq_with_well]]    

    return has_previous, has_relinq, relinq_with_well

def is_current(leases):
    return ~leases["Lease Effective Date"].isna() & leases["Lease Expiration Date"].isna()

def is_newly_available(leases, period):
    return leases["Lease Expiration Date"] == (period - 1)

def leases_to_backtest(leases, date_column, periods):
    for period in periods:
        # have to copy because we need to update expiry
        _leases = leases[leases[date_column] < period].copy()
        indx = _leases["Lease Expiration Date"] >= period
        _leases.loc[indx,"Lease Expiration Date"] = pd.NaT
        
        _leases["is_current"] = False
        _leases.loc[is_current(_leases), "is_current"] = True
        
        _leases["is_newly_available"] = False
        _leases.loc[is_newly_available(_leases, period), "is_newly_available"] = True
        
        yield _leases

def to_backtest(leases, date_column, periods):
    for period in periods:
        _leases = leases[leases[date_column] < period]
        yield _leases
        


def get_prod(lease, qdata, cumulative=False):
    if(lease in qdata.index):
        res = qdata.loc[lease,:]
        most_recent = res.index.max()
        if cumulative:
            oil_prod, gas_prod = \
                res.sum()[["Lease Oil Production (BBL)", 
                           "Lease Condensate Production (BBL)"]]
            water_depth = res["Lease Max Water Depth (meters)"].max()
            num_compl = res["Producing Completions"].max()
        else:
            oil_prod, gas_prod, water_depth, num_compl = \
                res.loc[most_recent, ["Lease Oil Production (BBL)", 
                                  "Lease Condensate Production (BBL)",
                                  "Lease Max Water Depth (meters)",
                                  "Producing Completions"]].values 
    else:
        most_recent = pd.NaT
        oil_prod, gas_prod, water_depth, num_compl = 0, 0, 0, 0

    return most_recent, oil_prod, gas_prod, water_depth, num_compl

def provide_current(test_area, leases, owners, qdata, cumulative=False):
    
    results = {}
    if test_area not in leases.index:
        results[("None","None")] = {"Company Name" : "None",
                        "Lease Code" : "None Available",
                        "Lease Effective Date" :  pd.NaT,
                        "Lease Expiration Date" :  pd.NaT,                              
                        "Oil Production" : 0, 
                        "Gas Production" : 0, 
                        "Water Depth" : 0, 
                        "Producing Completions" : 0 }
        results = pd.DataFrame(results.values(), index=results.keys())
        results.index.names = ("AREABLK", "Lease Number")
        return results
        
    res = leases.loc[test_area,:]
    indx = res["Lease Expiration Date"].isna()
    
    if np.any(indx) and res.index[indx][0] in owners.index:
        lease = res.index[indx][0]
        
        owner = owners.loc[lease, "Company Name"]


        
        most_recent, oil_prod, gas_prod, water_depth, num_compl = \
                get_prod(lease, qdata, cumulative=cumulative)
        results[(test_area, lease)] = {"Company Name" : owner,
                        "Lease Code" : res.loc[indx,
                                            "Lease Status Code"].values[0],
                        "Lease Effective Date" : \
                                 leases.loc[(test_area, lease),
                                        "Lease Effective Date"],
                        "Lease Expiration Date" : \
                                 leases.loc[(test_area, lease),
                                        "Lease Expiration Date"],
                        "Oil Production" : oil_prod, 
                        "Gas Production" : gas_prod, 
                        "Water Depth" : water_depth, 
                        "Producing Completions" : num_compl }        
    else:
        results[(test_area,"None")] = {"Company Name" : "None",
                        "Lease Code" : "NA",
                        "Lease Effective Date" :  pd.NaT,
                        "Lease Expiration Date" :  pd.NaT,                              
                        "Oil Production" : 0, 
                        "Gas Production" : 0, 
                        "Water Depth" : 0, 
                        "Producing Completions" : 0 }

    results = pd.DataFrame(results.values(), index=results.keys())
    results.index.names = ("AREABLK", "Lease Number")
    return results
        
    
def past_leases(test_area, leases, owners, qdata, cumulative=False):
    res = leases.loc[test_area]
    indx = ~res["Lease Expiration Date"].isna()
    res = res.loc[indx]

    if len(res) < 1:
        # no past leases
        results = {}
        results[(test_area,"None")] = {"Company Name" : "None",
                        "Lease Code" : "NA",
                        "Lease Effective Date" :  pd.NaT,
                        "Lease Expiration Date" :  pd.NaT,
                        "Oil Production" : 0, 
                        "Gas Production" : 0, 
                        "Water Depth" : 0, 
                        "Producing Completions" : 0 }
        results = pd.DataFrame(results.values(), index=results.keys())
        results.index.names = ("AREABLK", "Lease Number")
        
        return results

    
    results = {}
    for key,val in res.iloc[::-1].iterrows():
        most_recent, oil_prod, gas_prod, water_depth, num_compl = \
                get_prod(key, qdata, cumulative=True)
        
        if not most_recent:
            most_recent = val["Lease Expiration Date"]

        results[(test_area,key)] = {"Company Name" : "Unknown",
                        "Lease Code" : val["Lease Status Code"],
                        "Lease Effective Date" : val["Lease Effective Date"],
                        "Lease Expiration Date" : val["Lease Expiration Date"],
                        "Oil Production" : oil_prod, 
                        "Gas Production" : gas_prod, 
                        "Water Depth" : water_depth, 
                        "Producing Completions" : num_compl }

    results = pd.DataFrame(results.values(), index=results.keys())
    results.index.names = ("AREABLK", "Lease Number")
    return results
        

def read_curated_neighbourhoods(base_directory):

    nn_dtypes = {"Block Max Water Depth (meters)" : int}
    nn_dates = ["Lease Effective Date", "Lease Expiration Date"]

    nn = pd.read_csv(join(base_directory, "neighbourhood_leases.csv"), 
                     dtype=nn_dtypes,
                     parse_dates=nn_dates)
    nn["AREABLK_NN"] = nn["Area Code"].str.replace('\W{1,}','') + \
                       nn["Block Number"].str.replace('\W{1,}','')
    nn.drop(columns=["Area Code", "Block Number"], inplace=True)
    nn.sort_values(["AREABLK","AREABLK_NN"], inplace=True)
    nn.set_index(["AREABLK", "AREABLK_NN", "Lease Number"], inplace=True)
    nn.sort_values(["AREABLK_NN", "Lease Effective Date"],
                   ascending=False,inplace=True)
    nn.sort_index(level=0,inplace=True)

    return nn

def create_nn_data():
    """
    Expensive, so just make it and save it to disk. 
    """
    nn_dtypes = {"Block Max Water Depth (meters)" : int}
    nn_dates = ["Lease Effective Date", "Lease Expiration Date"]

    nn = pd.read_csv(join(base_directory, "neighbourhood_leases.csv"), 
                     dtype=nn_dtypes,
                     parse_dates=nn_dates)
    nn["AREABLK_NN"] = nn["Area Code"].str.replace('\W{1,}','') + \
                       nn["Block Number"].str.replace('\W{1,}','')
    nn.drop(columns=["Area Code", "Block Number"], inplace=True)
    nn.sort_values(["AREABLK","AREABLK_NN"], inplace=True)
    nn.set_index(["AREABLK", "AREABLK_NN", "Lease Number"], inplace=True)
    nn["Lease Effective Date"] = \
                    nn["Lease Effective Date"].dt.to_period(period_size)
    nn["Lease Expiration Date"] = \
                    nn["Lease Expiration Date"].dt.to_period(period_size)


def load_num_wells(base_directory, period_size="Q", code=None):
    """Shift Spud date by 1 period to simulate flow of information.
       Code "None" will mean any kind of well. Otherwise provide the 
       required codes.
    """
    wells_dates = ["Spud Date"]
    wells = pd.read_csv(join(base_directory, "Borehole.csv"),
                        parse_dates=wells_dates)
    wells["AREABLK"] = wells["Bottom Area"].str.replace("\W{1,}","") + \
                       wells["Bottom Block"].str.replace("\W{1,}","")
    wells.rename(columns={"Bottom Lease Number" : "Lease Number"}, inplace=True)
    wells["Spud Date"] = wells["Spud Date"].dt.to_period(period_size)
    # forward date the Spud Date to simulate flow of information:
    wells["Spud Date"] = wells["Spud Date"] + 1

    #
    if code is not None:
        wells = wells[wells["Type Code"] == code]

    wells.set_index(["AREABLK", "Lease Number", "Spud Date"], inplace=True)
    wells["NUMWELLS"] = 1
    wells.sort_index(level=2, inplace=True)
    
    # cumulative wells
    for key, val in wells.groupby(level=[0,1]):
        wells.loc[key, "NUMWELLS"] = val["NUMWELLS"].cumsum()
        
    wells.sort_index(level=[0,1,2], inplace=True)

    wells = wells[["Company Name","NUMWELLS", "True Vertical Depth (feet)"]]

    return wells
