import numpy as np
import plotly.graph_objects as go
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_daq as daq
import open3d as o3d
from dash.dependencies import Input, Output
import math
import copy

# DETAILS FOR APP
# WHEN YOU PRESS PROCEED TO FEEDBACK YOU CANT PRESS IT OR ROTATE AGAIN
# because it rotated the main pcl

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

source = o3d.io.read_point_cloud("01_balcony_mold_03.ply")
target = o3d.io.read_point_cloud("Ama_model_pc.ply")

voxel = 30

# function to translate the pcl1 so the centers of pcl1 and pcl2 are matched
# returns the translated pcl1
def match_centers(pcl1, pcl2):
    cen_pcl1 = pcl1.get_center()
    cen_pcl2 = pcl2.get_center()
    translation_vec = np.array(cen_pcl2 - cen_pcl1)
    pcl1.translate(translation_vec)
    return pcl1

# rotation functions, inputs pcl and degrees
# returns the rotated pcl
def rotate_Z(pcd, d):
    r = d*(math.pi/180) # convert it from degrees to radians
    zAxis = np.array((0, 0, r))
    zAxisArr = o3d.geometry.get_rotation_matrix_from_axis_angle(zAxis)
    print(zAxisArr)
    pcd.rotate(zAxisArr, True)
    return pcd
def rotate_Y(pcd, d):
    r = d*(math.pi/180) # convert it from degrees to radians
    yAxis = np.array((0, r, 0))
    yAxisArr = o3d.geometry.get_rotation_matrix_from_axis_angle(yAxis)
    print(yAxisArr)
    pcd.rotate(yAxisArr, True)
    return pcd
def rotate_X(pcd, d):
    r = d*(math.pi/180) # convert it from degrees to radians
    xAxis = np.array((r, 0, 0))
    xAxisArr = o3d.geometry.get_rotation_matrix_from_axis_angle(xAxis)
    print(xAxisArr)
    pcd.rotate(xAxisArr, True)
    return pcd


# function to create a figure of a point cloud
def create_cloud(cloud, downsample):
    downpcd = cloud.voxel_down_sample(voxel_size=int(downsample))
    array = np.asarray(downpcd.points)
    return go.Figure(data=[go.Scatter3d(x=array[:, 0], y=array[:, 1], z=array[:, 2],mode='markers', marker=dict(color='gray', size=1))])
# function to create a pcl trace, gray color
def create_trace_target(cloud, downsample):
    downpcd = cloud.voxel_down_sample(voxel_size=int(downsample))
    array = np.asarray(downpcd.points)
    return go.Scatter3d(x=array[:, 0], y=array[:, 1], z=array[:, 2],mode='markers', marker=dict(color='gray' , size=1))
# function to create a pcl trace, blue color
def create_trace_source(cloud, downsample):
    downpcd = cloud.voxel_down_sample(voxel_size=int(downsample))
    array = np.asarray(downpcd.points)
    return go.Scatter3d(x=array[:, 0], y=array[:, 1], z=array[:, 2],mode='markers', marker=dict(color='blue' , size=1))

def transf_from_rotation(dx, dy, dz):
    rz = dz * (math.pi / 180)  # convert it from degrees to radians
    zAxis = np.array((0, 0, rz))
    zAxisArr = o3d.geometry.get_rotation_matrix_from_axis_angle(zAxis)

    ry = dy * (math.pi / 180)
    yAxis = np.array((0, ry, 0))
    yAxisArr = o3d.geometry.get_rotation_matrix_from_axis_angle(yAxis)

    rx = dx * (math.pi / 180)
    xAxis = np.array((rx, 0, 0))
    xAxisArr = o3d.geometry.get_rotation_matrix_from_axis_angle(xAxis)

    rot_matrix = zAxisArr * yAxisArr * xAxisArr

    trans_matrix = np.zeros((4,4))
    trans_matrix[0,3]= 0
    trans_matrix[1,3]= 0
    trans_matrix[2, 3] = 0
    trans_matrix[3, 3] = 0
    trans_matrix[3, 0] = 0
    trans_matrix[3, 1] = 0
    trans_matrix[3, 2] = 0
    trans_matrix[3, 3] = 1
    trans_matrix[0:3, 0:3] = rot_matrix
    return trans_matrix

source_cent = match_centers(source, target)

fig = create_cloud(target, voxel)
fig.add_trace(create_trace_source(source_cent, voxel))
fig.update_layout(showlegend=False, scene_aspectmode='data', uirevision = True, scene = dict(xaxis = dict(title='', showbackground=False,showticklabels=False),yaxis = dict(title='', showbackground=False,showticklabels=False),zaxis = dict(title='', showbackground=False,showticklabels=False))),
#style=dict(display='flex', flexWrap='nowrap', width='2000', verticalAlignment='middle', margin='1px')
app.layout = html.Div([
    html.Div([
            html.Div([daq.Slider(id='slider_z', marks={'180':'Rotation on Z axis'}, min=1, max=360, step=1,value=180, size=400)]),
            html.Div([daq.Slider(id='slider_y', marks={'180':'Rotation on Y axis'}, min=1, max=360, step=1, value=180, size=400)]),
            html.Div([daq.Slider(id='slider_x', marks={'180':'Rotation on X axis'}, min=1, max=360, step=1, value=180, size=400)]),
    ],style={'width':'2000','display':'flex', 'flexWrap':'nowrap', 'align-items':'center', 'justify-content':'center'}),
#‘width’: ‘100%’, ‘display’: ‘flex’, ‘align-items’: ‘center’, ‘justify-content’: ‘center’
    html.Div(dcc.Graph(id='3d_scat', figure=fig), style={'height': '600', 'width':'2000', 'margin':'20px'}),
    html.Div([
        html.Div([html.Button('Proceed to Feedback', id='ICP', style={'height': 'auto'})]),
    ]),
])


# create a list with ids from sliders and a list with rotation values
ids = []
rot_val = []

@app.callback(Output('3d_scat', 'figure'),
            [Input('ICP', 'n_clicks'),
            Input('slider_z', 'value'),
            Input('slider_y', 'value'),
            Input('slider_x', 'value'),
            ])

def rotate(slider_z, slider_y, slider_x, ICP): # all the inputs from callback otherwise it breaks
    ctx = dash.callback_context

    if ctx.triggered:
        val = ctx.triggered[0]['value']
        # scale of sliders is 0-360
        # we map it to the scale of (-180)-180
        cur_value = (-180) + val
        cur_id = ctx.triggered[0]['prop_id'].split('.')[0]
        # it breaks if rotation is 0
        if cur_value == 0:
            cur_value = 1
        # appending the two arrays with the current id and value
        ids.append(cur_id)
        rot_val.append(cur_value)

    if (cur_id == 'slider_z'): # checks which button was pressed
        j = len(ids)-1 # length of the array without the last item which is the current one
        y_bool = True
        x_bool = True
        # loops backwards into the list with previous changes in sliders with ids
        # finds the latest changes in other axis and stores them
        while j >= 0:
            if y_bool and ids[j] == 'slider_y':
                    y_value = rot_val[j]
                    y_bool = False
                    print('y:', rot_val[j])
            if x_bool and ids[j] == 'slider_x':
                    x_value = rot_val[j]
                    x_bool = False
                    print('x:', rot_val[j])
            j = j - 1
        # clears data from graph
        fig.data = []
        # makes a copy of the pcl to rotate
        temp_pcl = copy.deepcopy(source_cent)
        # checks if there are previous changes in other axis
        if not y_bool:
            # rotate the copy with latest y value
            rotate_Y(temp_pcl, y_value)
        if not x_bool:
            # rotate the copy with the latest x value
            rotate_X(temp_pcl, x_value)
        # rotate the copy with the current value of z
        rotate_Z(temp_pcl, rot_val[len(rot_val)-1])
        print('z:', rot_val[len(rot_val)-1])
        # adds the trace of the scan and the model (both pcls)
        fig.add_trace(create_trace_target(target, voxel))
        fig.add_trace(create_trace_source(temp_pcl, voxel))
        return fig
    elif (cur_id == 'slider_y'):
        j = len(ids)-1
        z_bool = True
        x_bool = True
        while j >= 0:
            if z_bool and ids[j] == 'slider_z':
                    z_value = rot_val[j]
                    z_bool = False
            if x_bool and ids[j] == 'slider_x':
                    x_value = rot_val[j]
                    x_bool = False
            j = j - 1
        fig.data = []
        temp_pcl = copy.deepcopy(source_cent)
        if not z_bool:
            rotate_Z(temp_pcl, z_value)
        if not x_bool:
            rotate_X(temp_pcl, x_value)
        rotate_Y(temp_pcl, rot_val[len(rot_val)-1])
        fig.add_trace(create_trace_target(target, voxel))
        fig.add_trace(create_trace_source(temp_pcl, voxel))
        return fig
    elif (cur_id == 'slider_x'):
        j = len(ids)-1
        y_bool = True
        z_bool = True
        while j >= 0:
            if y_bool==True and ids[j] == 'slider_y':
                    y_value = rot_val[j]
                    y_bool = False
            if z_bool==False and ids[j] == 'slider_z':
                    z_value = rot_val[j]
                    z_bool = False
            j = j - 1
        fig.data = []
        temp_pcl = copy.deepcopy(source_cent)
        if not y_bool:
            rotate_Y(temp_pcl, y_value)
        if not z_bool:
            rotate_Z(temp_pcl, z_value)
        rotate_X(temp_pcl, rot_val[len(rot_val)-1])
        fig.add_trace(create_trace_target(target, voxel))
        fig.add_trace(create_trace_source(temp_pcl, voxel))
        return fig
    elif (cur_id == 'ICP'):
        z_bool = True
        y_bool = True
        x_bool = True
        j = len(ids)-1
        while j >= 0:
            if z_bool and ids[j]=='slider_z':
                z_bool = False
                z_value = rot_val[j]
            if y_bool and ids[j]=='slider_y':
                y_bool = False
                y_value = rot_val[j]
            if x_bool and ids[j]=='slider_x':
                x_bool = False
                x_value = rot_val[j]
            j = j - 1
        if not y_bool:
            rotate_Y(source_cent, y_value)
        if not z_bool:
            rotate_Z(source_cent, z_value)
        if not x_bool:
            rotate_X(source_cent, x_value)
        # compute the threshold for ICP based on average of nearest neighbor distance
        density_target = o3d.geometry.PointCloud.compute_nearest_neighbor_distance(target)
        threshold = (np.average(density_target)) * 0.009
        # threshold = 0.05 default value from open3D
        print("Threshold:", threshold)
        # a transformation matrix with rotation = 0, translation = 0 and scale = 1
        trans_init = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0],[0.0, 0.0, 0.0, 1.0]]
        # compute transformation from ICP
        reg_p2p = o3d.registration.registration_icp(
            source_cent, target, threshold, trans_init,
            o3d.registration.TransformationEstimationPointToPoint())

        # apply transformation to scan
        source_cent.transform(reg_p2p.transformation)
        #o3d.visualization.draw_geometries([source_cent, target])

        fig.data = []
        # adds the trace of the scan and the model (both pcls)
        fig.add_trace(create_trace_target(target, voxel))
        fig.add_trace(create_trace_source(source_cent, voxel))
        return fig





if __name__ == '__main__':
    app.run_server(debug=True)