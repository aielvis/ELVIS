import dash
import dash_table
import pandas as pd
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

from os.path import dirname, join

from elvis.io.boem_from_file import (load_num_wells,
                                     boem_leases,
                                     read_curated_neighbourhoods)

from elvis.visualization.utils import (load_dash_data,
                                       slice_well_by_area,
                                       convert_datetimes, slice_by_area,
                                       null_header, null_past, null_nn)
                                       
from elvis import datasets

base_directory = dirname(datasets.__file__)
freeze_data = join(base_directory, 'Freeze_Data\ 12_4_2019')
period_size = 'Q'

test_area = "WR970"

# ----- #
nn = read_curated_neighbourhoods(base_directory)
nn["Lease Effective Date"] = \
                nn["Lease Effective Date"].dt.to_period(period_size)
nn["Lease Expiration Date"] = \
                nn["Lease Expiration Date"].dt.to_period(period_size)

leases = boem_leases(base_directory)
leases.sort_index(inplace=True)
leases["Lease Effective Date"] = \
                leases["Lease Effective Date"].dt.to_period(period_size)
leases["Lease Expiration Date"] = \
                leases["Lease Expiration Date"].dt.to_period(period_size)
# drop bad data
leases.dropna(subset=["Lease Effective Date"], inplace=True)

wells = load_num_wells(base_directory)
print (wells)

leases = leases.join(wells, how="outer")
# ---- #

header, footer = slice_well_by_area(test_area, leases, nn)

# ---- #
convert_datetimes(header)
convert_datetimes(footer)

df = header
df3 = footer

app = dash.Dash(__name__)


app.layout = html.Div([
    dcc.Input(id='my-id', value=test_area, type='text'),
    html.H2("Current Lease"),
    
    dash_table.DataTable(        
        id='table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records')),
    
    html.H2("Area block neighbourhood"),            
    dash_table.DataTable(
        id='table3',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df3.to_dict('records')),
    
    ])


@app.callback(
    Output(component_id='table', component_property='data'),
    [Input(component_id='my-id', component_property='value')]
)                
def update_output_div(input_value):
    if input_value not in wells.index.get_level_values(0):
        return null_header


    header, footer = slice_well_by_area(input_value, leases, nn)
    convert_datetimes(header)
        
    return header.to_dict('records')


@app.callback(
    Output(component_id='table3', component_property='data'),
    [Input(component_id='my-id', component_property='value')]
)                     
def update_output_div3(input_value):    
    if input_value not in leases.index:
        return null_nn

    header, footer = slice_well_by_area(input_value, leases, nn)
    
    convert_datetimes(footer)
    
    return footer.to_dict('records')


if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
    
