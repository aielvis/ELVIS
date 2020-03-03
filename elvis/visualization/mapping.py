""" 
"""
from ipyleaflet import (basemap_to_tiles, basemaps, GeoJSON, Map, LayersControl,
                        TileLayer)

colors = {"equinor" : "#EC3E66",
          "chevron" : "#4E6CC9",
          "exxon"   : "#F12007",
          "shell"   : "#F39E0A"}

_boem_bsee = "https://gis.boem.gov/arcgis/rest/services/BOEM_BSEE/"
_boem_bath = "GOM_Deepwater_Bathymetry_and_Hillshade_Tiled/"
_map_server_suffix = "MapServer/tile/{z}/{y}/{x}"

tile_servers = {
    'boem_bathymetry' : {"url" : _boem_bsee + _boem_bath + _map_server_suffix,
                         "max_zoom" : 12}
}

center_of_gom = (26.9792212296875, -91.87030927187499)

def bathymetry_underlay(opacity=0.25):
    server = tile_servers["boem_bathymetry"]
    return basemap_to_tiles(server, opacity=opacity)

def geojson_underlay(geo_json, name, **style):
    return GeoJSON(data=geo_json,
                   name=name,
                   style=style)
    
def create_map_from_geojson(geo_interface,
                            underlays=None,
                            zoom=6,
                            default_tile_map=basemaps.Esri.WorldStreetMap,
                            color="green",
                            control=False,
                            opacity=0.5,
                            weight=0.1,
                            fillOpacity=0.5):
                            
    """ """
    
    m = Map(default_tiles=TileLayer(opacity=1.0),       
            layers=(basemap_to_tiles(default_tile_map,),),
            center=center_of_gom,
            zoom=zoom)
    
    _geo_json = GeoJSON(data=geo_interface, 
                        style = {'color': color, 
                                 'opacity':opacity, 
                                 'weight':weight, 
                                 'fillOpacity':fillOpacity})
    
    if isinstance(underlays, type([])):
        for u in underlays:
            m.add_layer(u)
    elif underlays is not None:
        m.add_layer(underlays)

    # add overlay data
    m.add_layer(_geo_json)

    if control:
        m.add_control(LayersControl())
        

    return m

