

import dash
import dash_table
import pandas as pd
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

from os.path import dirname, join
from elvis.visualization.utils import (load_dash_data,
                                       convert_datetimes, slice_by_area,
                                       null_header, null_past, null_nn)
                                       
from elvis import datasets

base_directory = dirname(datasets.__file__)
freeze_data = join(base_directory, 'Freeze_Data\ 12_4_2019')

test_area = "MC257"
test_area = "MC300"
test_area = "WC17"
test_area = "WR271"

leases, owners, qdata, bid_data, winning_bids, consortia, nn = load_dash_data(
                test_area, base_directory, freeze_data)


header, block_past_leases, _nn = slice_by_area(test_area, leases, owners, qdata,
                  bid_data, winning_bids, consortia, nn)
convert_datetimes(header)
convert_datetimes(block_past_leases)


df = header
df2 = block_past_leases
df3 = _nn

app = dash.Dash(__name__)


def update_df(_df):
    df = _df
    

app.layout =html.Div([
    dcc.Input(id='my-id', value=test_area, type='text'),
    html.H2("Current Lease"),    
    dash_table.DataTable(        
        id='table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records')),
    html.H2("Previous Lease(s)"),        
    dash_table.DataTable(
        id='table2',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records')),
    html.H2("Area block neighbourhood"),            
    dash_table.DataTable(
        id='table3',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records')),
    
    ])

@app.callback(
    Output(component_id='table', component_property='data'),
    [Input(component_id='my-id', component_property='value')]
)                
def update_output_div(input_value):
    if input_value not in leases.index:
        return null_header

    header, block_past_leases, _nn = slice_by_area(input_value,
                                leases, owners, qdata,
                  bid_data, winning_bids, consortia, nn)    
    convert_datetimes(header)
    return header.to_dict('records')

@app.callback(
    Output(component_id='table2', component_property='data'),
    [Input(component_id='my-id', component_property='value')]
)                     
def update_output_div2(input_value):
    if input_value not in leases.index:
        return null_past
    
    header, block_past_leases, _nn = slice_by_area(input_value,
                                leases, owners, qdata,
                  bid_data, winning_bids, consortia, nn)    
    convert_datetimes(block_past_leases)
    return block_past_leases.to_dict('records')

@app.callback(
    Output(component_id='table3', component_property='data'),
    [Input(component_id='my-id', component_property='value')]
)                     
def update_output_div3(input_value):
    if input_value not in leases.index:
        return null_nn
    
    header, block_past_leases, _nn = slice_by_area(input_value,
                                leases, owners, qdata,
                  bid_data, winning_bids, consortia, nn)    
    convert_datetimes(_nn)
    return _nn.to_dict('records')

if __name__ == '__main__':
    app.run_server(debug=True, port=8053)
    
