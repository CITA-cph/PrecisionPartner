import json
import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Input, Output, State
import vector3d
import math
from scipy.cluster.vq import kmeans
import base64
import os
import datetime

#VARIABLES_____________________________________________________________________________________________
#Dash related
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True    #IMPORTANT FOR MULTIPAGE APPS


#Input from grasshopper based on the digital 3d model
pathDimCloud1 = "cloud1.xyz"   #start points of dimensions
pathDimCloud2 = "cloud2.xyz"   #end points of dimensions
red = [1, 0, 0]                #color assigned to them

#Input from scanner based on scanned geometry
pathScanCloud = "RoyalT_faketrackers.ply"
grey = [0.5, 0.5, 0.5]
downSize = 20

#Search parameters
searchRadius = 20

#Input mesh for vizualization
mesh = o3d.io.read_triangle_mesh("musikmeshSINGLE.ply")

#input for dim tolerance
tolerance = 3
subValue_0 = 20
toleranceSelect_0 = 3

resultsX = []
resultsY = []
resultsZ = []
resultsID = []

#image display
image_filename = os.path.join(os.getcwd(), 'gradient.png')
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

logos_filename = os.path.join(os.getcwd(), 'logos2.png')
encoded_logos = base64.b64encode(open(logos_filename, 'rb').read())

#Seperate kdtree creation from cloud parsing
def loadCloud(path, color,bool):
    # Cloud path, color for display, bool to make kd tree

    # load cloud for nearest point search
    cloud = o3d.io.read_point_cloud(path)
    print(cloud, "has been loaded successfully")
    cloud.paint_uniform_color(color)

    #create KDtree
    if bool == True:
        tree = o3d.geometry.KDTreeFlann(cloud)
        print("KDtree has been created")
        return cloud, tree
    else:
        return cloud

#CloudViz
def create_cloud(cloud, downsample):
    #cloud that has already been loaded and lives in the directory
    downpcd = cloud.voxel_down_sample(voxel_size=int(downsample))
    array = np.asarray(downpcd.points)
    print("light cloud metadata created")

    return go.Figure(data=[go.Scatter3d(x=array[:, 0], y=array[:, 1], z=array[:, 2],mode='markers', marker=dict(color='gray' , size=1))])


#PUSH TO INTERFACE___________________________________________________________________________________________________________
print("initializing UI")

app.layout = html.Div([
    html.Div([
            html.P("Precision Partner_Dimension Feedback", style={ 'fontFamily':'dosis','fontSize':'30px'})]),

    html.Div([
        dcc.Tabs(id='tabs-example', style={ 'fontFamily':'dosis','fontSize':'15px'}, value='tab-1', children=[
            dcc.Tab(label='1. Upload and automatic alignment',style={ 'fontFamily':'dosis','fontSize':'15px'}, value='tab-1'),
            dcc.Tab(label='2. Manual alignment',style={ 'fontFamily':'dosis','fontSize':'15px'}, value='tab-2'),
            dcc.Tab(label='3. Feedback',style={ 'fontFamily':'dosis','fontSize':'15px'},  value='tab-3'),
    ]),
    html.Div(id='tabs-example-content')
])])

#Change Tabs
@app.callback(Output('tabs-example-content', 'children'),
              [Input('tabs-example', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
                        html.Div([dcc.Upload(id='upload-data', children=html.Div([html.A('Upload Files')]), style={'height': 'auto', 'margin': '10px', 'display': 'inline-block'}, multiple=True)]),
                        html.Div(id='upload-output')
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H3('Please follow the steps')
        ])

    elif tab == 'tab-3':
        return html.Div([
            html.H3('Please follow the steps')
        ])

def parse_contents(contents, filename, date):
    #Unserialize the content of the txt file
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    print(decoded)

    p = str(decoded)
    p = p[2:-1]
    print(p)

    pathScanCloud = p + "/RoyalT_faketrackers.ply"
    pcd, pcd_tree = loadCloud(pathScanCloud, grey, True)
    c = create_cloud(pcd, downSize)
    c.update_layout(scene_aspectmode='data', showlegend=False, uirevision=True,
                    scene=dict(xaxis=dict(title='', showbackground=False, showticklabels=False),
                               yaxis=dict(title='', showbackground=False, showticklabels=False),
                               zaxis=dict(title='', showbackground=False, showticklabels=False)), width=800,
                    height=450, margin=dict(r=0, l=0, b=10, t=10)),


    #load mesh
    #run sampling pioints
    #run Ransac ICP
    #display trace
    #ask for user confirmation
    #switch to next tab

    return html.Div(dcc.Graph(id='DemoView', figure= c), style={'height': 'auto'})

@app.callback(Output('upload-output', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [parse_contents(c, n, d) for c, n, d in zip(list_of_contents, list_of_names, list_of_dates)]
        return children


if __name__ == '__main__':
    app.run_server(debug=True)
