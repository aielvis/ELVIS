

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

mlot = pd.read_csv(join(base_directory, "model-1", "pred-rf-aug19.csv"))
print (mlot.columns)
mlot.drop(columns=['Unnamed: 0'], inplace=True)


print ("")
df = mlot

app = dash.Dash(__name__)

    

app.layout =html.Div([
    dash_table.DataTable(        
        id='table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records')),
    ])

if __name__ == '__main__':
    app.run_server(debug=True, port=9000)
    
