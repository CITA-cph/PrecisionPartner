import json
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
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
import dash_bootstrap_components as dbc
import copy

#notes
# move global variables at the start of the code
# start tabs with button yes and no
# icp in line 120, turn it off for faster debugging
# currently loading the model twice line 71 and 503 instead DCC STORE (also the scan)
# declare filenames at the start of the code


#VARIABLES_____________________________________________________________________________________________
#Dash related
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions = True    #IMPORTANT FOR MULTIPAGE APPS


#Input from grasshopper based on the digital 3d model

red = [1, 0, 0]                #color assigned to them

#Input from scanner based on scanned geometry

grey = [0.5, 0.5, 0.5]
downSize = 20

#Search parameters
searchRadius = 20

#input for dim tolerance
tolerance = 3
subValue_0 = 20
toleranceSelect_0 = 3

resultsX = []
resultsY = []
resultsZ = []
resultsID = []

# arrays for tab 2 rotation with sliders
ids = []
rot_val = []

#image display
logos_filename = os.path.join(os.getcwd(), 'logos2.png')
encoded_logos = base64.b64encode(open(logos_filename, 'rb').read())

#Iliana
def loadCloud(path, color):
    # Cloud path, color for display, bool to make kd tree

    # load cloud for nearest point search
    cloud = o3d.io.read_point_cloud(path)
    print(cloud, "has been loaded successfully")
    cloud.paint_uniform_color(color)

    return cloud

def loadModel_samplePts(path, scan): # input also scan to compare densities
    model_mesh = o3d.io.read_triangle_mesh(path)
    # find the bbox volumes to compare
    bbox_scan = scan.get_oriented_bounding_box()
    bbox_model = model_mesh.get_oriented_bounding_box()
    num_pts_scan = np.asarray(scan.points)
    print("--Number of pts in scan: ", np.size(num_pts_scan, 0))
    num_pts_model = (o3d.geometry.OrientedBoundingBox.volume(bbox_model) * np.size(num_pts_scan, 0)) \
                    / o3d.geometry.OrientedBoundingBox.volume(bbox_scan)
    print("--Number of pts in model: ", int(num_pts_model))

    model = model_mesh.sample_points_uniformly(int(num_pts_model))
    return model

def preprocess_pcd_forRANSAC(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size=int(voxel_size))

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def ransac_icp(source, target):
    # find the voxel size based on average nearest neighbor distance of the scan
    density_source = o3d.geometry.PointCloud.compute_nearest_neighbor_distance(source)
    voxel_size = 67 * (np.average(density_source))
    voxel_size = np.around(voxel_size, decimals=0)

    source_down, source_fpfh = preprocess_pcd_forRANSAC(source, voxel_size)
    target_down, target_fpfh = preprocess_pcd_forRANSAC(target, voxel_size)

    distance_threshold = voxel_size * 1.5
    print("   RANSAC registration on downsampled point clouds.")
    print("   distance threshold %.3f." % distance_threshold)
    result_ransac = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))
    distance_threshold2 = voxel_size * 0.4
    print("   Refine registration")
    print("   Point-to-plane ICP registration with distance threshold %.3f." % distance_threshold2)
#    result_icp = o3d.registration.registration_icp(
#        source, target, distance_threshold2, result_ransac.transformation,
#        o3d.registration.TransformationEstimationPointToPoint())
#    source.transform(result_icp.transformation)
    return source

# rotate functions for tab 2
def rotate_Z(pcd, d):
    r = d*(math.pi/180) # convert it from degrees to radians
    zAxis = np.array((0, 0, r))
    zAxisArr = o3d.geometry.get_rotation_matrix_from_axis_angle(zAxis)
    pcd.rotate(zAxisArr, True)
    return pcd
def rotate_Y(pcd, d):
    r = d*(math.pi/180) # convert it from degrees to radians
    yAxis = np.array((0, r, 0))
    yAxisArr = o3d.geometry.get_rotation_matrix_from_axis_angle(yAxis)
    pcd.rotate(yAxisArr, True)
    return pcd
def rotate_X(pcd, d):
    r = d*(math.pi/180) # convert it from degrees to radians
    xAxis = np.array((r, 0, 0))
    xAxisArr = o3d.geometry.get_rotation_matrix_from_axis_angle(xAxis)
    pcd.rotate(xAxisArr, True)
    return pcd

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

#Gabriella
#CloudViz
def create_cloud(cloud, downsample, color):
    #cloud that has already been loaded and lives in the directory
    downpcd = cloud.voxel_down_sample(voxel_size=int(downsample))
    array = np.asarray(downpcd.points)

    return go.Figure(data=[go.Scatter3d(x=array[:, 0], y=array[:, 1], z=array[:, 2],mode='markers', marker=dict(color=color , size=1))])

def create_cloud_trace(pcd, downsample, color): # returns dash trace
    downpcd = pcd.voxel_down_sample(voxel_size=int(downsample))
    array = np.asarray(downpcd.points)
    meshX = array[:, 0]
    meshY = array[:, 1]
    meshZ = array[:, 2]
    #print("cloud dimension points created")
    return go.Scatter3d(x=meshX, y=meshY, z=meshZ, mode='markers', marker=dict(color=color, size=1))

def create_mesh(mesh):
    #3d mesh that lives in directory
    mesh_pts = np.asarray(mesh.vertices)
    mesh_tris = np.asarray(mesh.triangles)
    print("mesh metadata created")
    return go.Figure(data=[go.Mesh3d(name='3D Model', x = mesh_pts[:, 0], y = mesh_pts[:, 1], z = mesh_pts[:, 2], i=mesh_tris[:,0], j=mesh_tris[:,1], k=mesh_tris[:,2], color='gray', opacity=0.85)])

def createMeshDimPoints(cloud):
    #Start/End Point Cloud with the standard dimension points exported from grasshopper
    array = np.asarray(cloud.points)  #because we are loading the cloud as xyz
    meshX = array[:, 0]
    meshY = array[:, 1]
    meshZ = array[:, 2]
    print("mesh dimension points created")
    return array, go.Scatter3d(x=meshX, y=meshY, z=meshZ, mode='markers', marker=dict(color='red', size=3))

def create_dimLines(array1,array2):
    #array of start points, array of end points

    #to display lines properly, the coordinates must be interwoven
    xArray1 = array1[:, 0]
    xArray2 = array2[:, 0]
    ptX = np.empty((xArray1.size + xArray2.size), dtype = xArray1.dtype)
    ptX[0::2] = xArray1
    ptX[1::2] = xArray2
    ptX.tolist()
    #print(ptX)

    yArray1 = array1[:, 1]
    yArray2 = array2[:, 1]
    ptY = np.empty((yArray1.size + yArray2.size), dtype = yArray1.dtype)
    ptY[0::2] = yArray1
    ptY[1::2] = yArray2
    ptY.tolist()
    #print(ptY)

    zArray1 = array1[:, 2]
    zArray2 = array2[:, 2]
    ptZ = np.empty((zArray1.size + zArray2.size), dtype = zArray1.dtype)
    ptZ[0::2] = zArray1
    ptZ[1::2] = zArray2
    ptZ.tolist()
    #print(ptZ)

    lines= []
    #Create a trace for each line, points 2 by 2
    for i in range(len(ptX)-1):
        xlinePt = np.array([ptX[i], ptX[i+1]])
        ylinePt = np.array([ptY[i], ptY[i+1]])
        zlinePt = np.array([ptZ[i], ptZ[i+1]])

        #Show every second line
        if i % 2 == 0:
            lines.append(go.Scatter3d(x=xlinePt, y=ylinePt, z=zlinePt, mode="lines", marker=dict(color='red', size=6)))

    print("dimension lines created")
    return lines

def create_dimText_mesh(array1, array2):
    # array of start points, array of end points, figure to apply to

    #locate midpoints (location for text display)
    midpointsX = ((array2[:, 0]) + (array1[:, 0])) / 2
    midpointsY = ((array2[:, 1]) + (array1[:, 1])) / 2
    midpointsZ = ((array2[:, 2]) + (array1[:, 2])) / 2
    print("midpoints have been calculated")

    #find cloud centroid
    rawCentroid = kmeans(np.asarray(pcd.points), 1)
    centroid = rawCentroid[0]
    centX = centroid[:, 0]
    centY = centroid[:, 1]
    centZ = centroid[:, 2]

    #Calculate vector between center and midpoints
    vecX = (midpointsX) - (centX)
    vecY = (midpointsY) - (centY)
    vecZ = (midpointsZ) - (centZ)

    #Calculate Vector Amplitude
    VcompX = vecX ** 2
    VcompY = vecY ** 2
    VcompZ = vecZ ** 2
    Vsum = VcompX + VcompY + VcompZ
    Vmag = np.sqrt(Vsum)

    #Calculate Unit vector
    uVecX = vecX / Vmag
    uVecY = vecY / Vmag
    uVecZ = vecZ / Vmag

    #Move midpoints by unit vector
    dispX = midpointsX + (uVecX * 150)
    dispY = midpointsY + (uVecY * 150)
    dispZ = midpointsZ
    print("midpoints have been offset")

    # Calculate distance
    compX = ((array2[:, 0]) - (array1[:, 0])) ** 2
    compY = ((array2[:, 1]) - (array1[:, 1])) ** 2
    compZ = ((array2[:, 2]) - (array1[:, 2])) ** 2
    sum = compX + compY + compZ
    dist = np.sqrt(sum)
    global distN_mesh
    distN_mesh = np.round(dist, 0)
    print("distances have been calculated")

    return distN_mesh, go.Scatter3d(x=dispX, y=dispY, z=dispZ, mode='text', text=distN_mesh, textposition='middle center', textfont=dict(color="red"))
    #if we want to make different colours, it needs to be here, create a list of go.scatter like in lines

def create_dimText_cloud(array1, array2, distN_mesh,value):
    # array of start points, array of end points, figure to apply to

    #locate midpoints (location for text display)
    midpointsX = ((array2[:, 0]) + (array1[:, 0])) / 2
    midpointsY = ((array2[:, 1]) + (array1[:, 1])) / 2
    midpointsZ = ((array2[:, 2]) + (array1[:, 2])) / 2
    print("midpoints have been calculated")

    #find cloud centroid
    rawCentroid = kmeans(np.asarray(pcd.points), 1)
    centroid = rawCentroid[0]
    centX = centroid[:, 0]
    centY = centroid[:, 1]
    centZ = centroid[:, 2]

    #Calculate vector between center and midpoints
    vecX = (midpointsX) - (centX)
    vecY = (midpointsY) - (centY)
    vecZ = (midpointsZ) - (centZ)

    #Calculate Vector Amplitude
    VcompX = vecX ** 2
    VcompY = vecY ** 2
    VcompZ = vecZ ** 2
    Vsum = VcompX + VcompY + VcompZ
    Vmag = np.sqrt(Vsum)

    #Calculate Unit vector
    uVecX = vecX / Vmag
    uVecY = vecY / Vmag
    uVecZ = vecZ / Vmag

    #Move midpoints by unit vector
    dispX = midpointsX + (uVecX * 150)
    dispY = midpointsY + (uVecY * 150)
    dispZ = midpointsZ
    print("midpoints have been offset")

    # Calculate distance
    compX = ((array2[:, 0]) - (array1[:, 0])) ** 2
    compY = ((array2[:, 1]) - (array1[:, 1])) ** 2
    compZ = ((array2[:, 2]) - (array1[:, 2])) ** 2
    sum = compX + compY + compZ
    dist = np.sqrt(sum)
    distN_cloud = np.round(dist, 0)
    print("distances have been calculated")

    distPurple = []
    distOrange = []
    distRed = []

    coorX = []
    coorY = []
    coorZ = []
    print(len(distN_cloud))
    for i in range(len(distN_cloud)):

        diff = distN_cloud[i] - distN_mesh[i]
        curr = np.round(distN_cloud[i],0)
        coorX.append(dispX[i])
        coorY.append(dispY[i])
        coorZ.append(dispZ[i])

        if diff <= -(value) :
            distPurple.append(go.Scatter3d(x=coorX, y=coorY, z=coorZ, mode='text', text=int(curr), textposition='middle center', textfont=dict(color="darkmagenta")))
            print("added to purple")
        if distN_cloud[i] - distN_mesh[i] >= (value):
            distOrange.append(go.Scatter3d(x=coorX, y=coorY, z=coorZ, mode='text', text=int(curr), textposition='middle center', textfont=dict(color="darkorange")))
            print("added to orange")
        if -(value - 1) <= distN_cloud[i] - distN_mesh[i] <= (value - 1):
            distRed.append(go.Scatter3d(x=coorX, y=coorY, z=coorZ, mode='text', text=int(curr), textposition='middle center',textfont=dict(color="red")))
            print("added to red")
        coorX.clear()
        coorY.clear()
        coorZ.clear()

    allDistCloud = distPurple + distOrange + distRed

    return allDistCloud

def runSearch(ScanCloud,DimCloud):
    #KD tree to search in, Cloud that defines anchor points
    tree = o3d.geometry.KDTreeFlann(ScanCloud)
    print("KDtree has been created")


    #Perform search
    closestPtIdxList = []
    i = 0
    for p in DimCloud.points:
        searchOutput = [k, idx, dist] = tree.search_knn_vector_3d(DimCloud.points[i], 2)
        #first point is point itself. First tuple item is how many points found, second is their index in the cloud, the third is the distance to anchor
        indexes = searchOutput[1]
        closestPtIdx = indexes[0]  # we can use idx 0 here because we are searching in another cloud
        #pcd.colors[closestPtIdx] = [0, 0, 1]  # colour it in blue #ILIANA comment it error
        closestPtIdxList.append(closestPtIdx)
        i += 1
    print("Indexes of cloud closest points found",closestPtIdxList)

    #Convert to results to numpy array
    pList = []
    for i in range(len(closestPtIdxList)):
        p = ScanCloud.points[closestPtIdxList[i]]
        pList.append(p)
    pArray = np.row_stack(pList)    #this creates an array (15, 3) of pointcloud coordinates p1 for standard dimensions
    #np.savetxt("Cloudp1.txt",pArray, fmt= '%f')
    print("points have been stored in numpy array")

    return pArray


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
    html.Div([html.P("Are the two point clouds aligned?", style={'height': 'auto', 'display': 'inline'}),
              html.Button('NO', id='button-not', style={'height': 'auto'}),
              html.Button('YES', id='button-yes', style={'height': 'auto'})
              ], style=dict(display='none')),
    html.Div([dcc.Loading(id='tabs-example-content')]) # DIF
])])

#Change Tabs
@app.callback(Output('tabs-example-content', 'children'),
              [Input('tabs-example', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
                        html.Div([dcc.Upload(id='upload-data', children=html.Div([html.A('Upload Files')]), style={'height': 'auto', 'margin': '10px', 'display': 'inline-block'}, multiple=True)]),
                        html.Div(id='upload-output'),
                        html.Div([html.P("Are the two point clouds aligned?", style={'height': 'auto', 'display': 'inline'}),
                                  html.Button('NO', id='button-not', style={'height': 'auto'}),
                                  html.Button('YES', id='button-yes', style={'height': 'auto'})
                                  ], style=dict(display='flex', flexWrap='nowrap', width=2000, verticalAlignment='middle')),
        ])
    elif tab == 'tab-2':
        if readyTab3 == True:
            return html.Div([
                html.Div([html.Button('Start', id='start-rotate', style={'height': 'auto', 'margin': '3px'})]),
                html.Div(id='rotate')
                    ])
        else:
            return html.Div([
                    html.H3('Please follow the steps')
                ])

    elif tab == 'tab-3':
            if readyTab3 == True:

                return html.Div([
                    html.Div([html.Button('Start', id='start-dim', style={'height': 'auto', 'margin': '3px'})]),
                    html.Div(id='dimension-feedback')
                ])

            else:
                return html.Div([
                    html.H3('Please follow the steps')
                ])


def parse_contents(contents, filename, date):
    #Unserialize the content of the txt file
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)

    global p
    p = str(decoded)
    p = p[2:-1]
    print(p)

    global pcd
    global scan_pcd

    #Load model as a mesh
    pathModel = p + "/Amabrogade_model.ply"
    pathScanCloud = p + "/Ama_mold_scan.ply"
    # load the scan as a point cloud
    scan_pcd = loadCloud(pathScanCloud, red)
    # load the mesh and sample points based on scan's density
    pcd = loadModel_samplePts(pathModel, scan_pcd)
    c = create_cloud(pcd, downSize, 'gray')
    c.update_layout(scene_aspectmode='data', showlegend=False, uirevision=True,
                    scene=dict(xaxis=dict(title='', showbackground=False, showticklabels=False),
                               yaxis=dict(title='', showbackground=False, showticklabels=False),
                               zaxis=dict(title='', showbackground=False, showticklabels=False)), width=800,
                    height=450, margin=dict(r=0, l=0, b=10, t=10)),

    # run icp and add the scan as a trace
    scan_pcd = ransac_icp(scan_pcd, pcd)
    c.add_trace(create_cloud_trace(scan_pcd, downSize, 'red'))

    #activate the thrid tab
    global readyTab3
    readyTab3 = True

    return html.Div(dcc.Graph(id='DemoView', figure= c), style={'height': 'auto'})


#Runs Ransac ICP when the file is uploaded
@app.callback(Output('upload-output', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [parse_contents(c, n, d) for c, n, d in zip(list_of_contents, list_of_names, list_of_dates)]
        return children



def createTab3(p, scan_pcd):
    #load gradient image
    image_filename = p + '/gradient.png'
    encoded_image = base64.b64encode(open(image_filename, 'rb').read())

    #load mesh
    pathMesh = p + "/Amabrogade_model.ply"
    mesh = o3d.io.read_triangle_mesh(pathMesh) # we do that in function loadModel_samplePts

    #load standard dimension cloud
    pathDimCloud1 = p + '/cloud1.xyz'   #start
    pathDimCloud2 = p + '/cloud2.xyz'   #end

    d1pcd = loadCloud(pathDimCloud1, red)
    d2pcd = loadCloud(pathDimCloud2, red)

    cldStartPtArray = runSearch(scan_pcd, d1pcd)
    cldPEndPtArray = runSearch(scan_pcd, d2pcd)

    meshStartpts, dashInfoStart = createMeshDimPoints(d1pcd)
    meshEndpts, dashInfoEnd = createMeshDimPoints(d2pcd)

    m = create_mesh(mesh)
    m.add_trace(dashInfoStart)  # Create start points
    m.add_trace(dashInfoEnd)  # Create end points
    lines_mesh = create_dimLines(meshStartpts, meshEndpts)  # create dimension lines
    meshDimensionText = create_dimText_mesh(meshStartpts, meshEndpts)[1]
    m.add_trace(meshDimensionText)  # Create dimension text
    for item in lines_mesh:
        m.add_trace(item)
    m.update_layout(scene_aspectmode='data', showlegend=False, uirevision=True,
                    scene=dict(xaxis=dict(title='', showbackground=False, showticklabels=False),
                               yaxis=dict(title='', showbackground=False, showticklabels=False),
                               zaxis=dict(title='', showbackground=False, showticklabels=False)), width=800, height=450,
                    margin=dict(r=0, l=0, b=10, t=10)),

#
    return html.Div([html.P("Select tolerance: ", style={'height': 'auto', 'display': 'inline', 'fontFamily': 'calibri'}),
                     html.Div([dcc.Dropdown(id='toleranceSelect',options=[{'label': '+/- 2', 'value': '2'}, {'label': '+/- 3', 'value': '3'}, {'label': '+/- 5', 'value': '5'}, {'label': '+/- 10', 'value': '10'}], value='2',style={'width': 200, 'height': 'auto', 'margin': '3px'})])
    ], style=dict(display='flex', flexWrap='nowrap', width=2000, verticalAlignment='middle')),\
           html.Div([html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()))], style=dict(display='flex', flexWrap='nowrap', width=2000, verticalAlignment='middle')), \
           html.Div([
               html.Div(id='empty', style={'width': 155, 'height': 'auto', 'display': 'inline'}),
               html.Div(id='neg', style={'width': 170, 'height': 'auto', 'display': 'inline', 'fontFamily': 'calibri'}),
               html.Div(id='pos', style={'height': 'auto', 'display': 'inline', 'fontFamily': 'calibri'})],
               style=dict(display='flex', flexWrap='nowrap', width=2000, verticalAlignment='middle')),\
           html.Div([
                    html.Div(id='empty2', style={'width': 20, 'height': 'auto', 'display': 'inline'}),
                    html.P("3d Model ", style={'height': 'auto', 'display': 'inline', 'fontFamily': 'dosis', 'fontSize': '20px'})],
    style=dict(display='flex', flexWrap='nowrap', width=2000, verticalAlignment='middle')),\
           html.Div([
                    html.Div(dcc.Graph(id='3d_mesh', figure= m), style={'height': 'auto'}),
    ], style=dict(display='flex', flexWrap='nowrap', width='99%', verticalAlignment='middle'))



#Runs dimension feedback when start is pressed
@app.callback(Output('dimension-feedback', 'children'),
                [Input('start-dim', 'n_clicks')])
def showDims(n_clicks):
    ctx = dash.callback_context

    if not ctx.triggered:
        nothingYet = 'no id yet'
    else:
        ids = ctx.triggered

        name = ids[0]['prop_id'].split('.')[0]

        if name == 'start-dim':
            print("yes")
            children = createTab3(p,scan_pcd)

            return children

#These two change the dim tolerance gradient annotation
@app.callback(Output('neg', 'children'),
            [Input('toleranceSelect', 'value')])
def update_neg(value):
    return ' -{}   '.format(value)

@app.callback(Output('pos', 'children'),
            [Input('toleranceSelect', 'value')])
def update_pos(value):
    return ' +{} '.format(value)

def createTab2(pcd, scan_pcd):
    # global var in the first lines
    global fig
    fig = create_cloud(pcd, downSize, 'grey')
    fig.add_trace(create_cloud_trace(scan_pcd, downSize, 'red'))
    fig.update_layout(scene_aspectmode='data', uirevision=True,
                      scene=dict(xaxis=dict(title='', showbackground=False, showticklabels=False),
                                 yaxis=dict(title='', showbackground=False, showticklabels=False),
                                 zaxis=dict(title='', showbackground=False, showticklabels=False)), width=1000,
                      height=500, margin=dict(r=0, l=0, b=10, t=10)),
    print("tab 2 is made")
    return html.Div([
    html.Div([
            html.Div([daq.Slider(id='slider_z', marks={'180':'Rotation on Z axis'}, min=1, max=360, step=1,value=180, size=400)]),
            html.Div([daq.Slider(id='slider_y', marks={'180':'Rotation on Y axis'}, min=1, max=360, step=1, value=180, size=400)]),
            html.Div([daq.Slider(id='slider_x', marks={'180':'Rotation on X axis'}, min=1, max=360, step=1, value=180, size=400)]),
    ],style={'width':'2000','display':'flex', 'flexWrap':'nowrap', 'align-items':'center', 'justify-content':'center'}),
    html.Div(id='rot_3d_scat'),
    html.Div(dcc.Graph(id='3d_scat', figure=fig), style={'height': '600', 'width':'2000', 'margin':'20px'}),
    html.Div([
        html.Div([html.Button('Proceed to Feedback', id='ICP', style={'height': 'auto'})]),
    ])
])


@app.callback(Output('rotate', 'children'),
    [Input('start-rotate', 'n_clicks')])
def activate_tab2(clicks):
    ctx = dash.callback_context

    if not ctx.triggered:
        nothingYet = 'no id yet'
    else:
        ids = ctx.triggered

        name = ids[0]['prop_id'].split('.')[0]

        if name == 'start-rotate':
            children = createTab2(pcd, scan_pcd)

            return children

@app.callback(Output('rot_3d_scat', 'children'),
            [Input('ICP', 'n_clicks'),
            Input('slider_z', 'value'),
            Input('slider_y', 'value'),
            Input('slider_x', 'value'),
            ])
def rotate_plot(icp, slider_z, slider_y, slider_x):
    ctx = dash.callback_context
    new_fig = create_cloud(pcd, downSize, 'grey')
    new_fig.add_trace(create_cloud_trace(scan_pcd, downSize, 'red'))
    new_fig.update_layout(scene_aspectmode='data', uirevision=True,
                      scene=dict(xaxis=dict(title='', showbackground=False, showticklabels=False),
                                 yaxis=dict(title='', showbackground=False, showticklabels=False),
                                 zaxis=dict(title='', showbackground=False, showticklabels=False)), width=1000,
                      height=500, margin=dict(r=0, l=0, b=10, t=10)),

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
        print("id",cur_id, "value", cur_value)
    else:
        cur_id = 'no id yet'
    print(rotate(icp, slider_z, slider_y, slider_x, new_fig, cur_id))
    return rotate(icp, slider_z, slider_y, slider_x, new_fig, cur_id)

def rotate(icp, slider_z, slider_y, slider_x, new_fig, cur_id): # all the inputs from callback otherwise it breaks

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
            if x_bool and ids[j] == 'slider_x':
                    x_value = rot_val[j]
                    x_bool = False
            j = j - 1
        # clears data from graph
        new_fig.data = []
        # makes a copy of the pcl to rotate
        temp_pcl = copy.deepcopy(scan_pcd)
        # checks if there are previous changes in other axis
        if not y_bool:
            # rotate the copy with latest y value
            temp_pcl = rotate_Y(temp_pcl, y_value)
        if not x_bool:
            # rotate the copy with the latest x value
            temp_pcl = rotate_X(temp_pcl, x_value)
        # rotate the copy with the current value of z
        temp_pcl = rotate_Z(temp_pcl, rot_val[len(rot_val)-1])
        # adds the trace of the scan and the model (both pcls)
        new_fig.add_trace(create_cloud_trace(pcd, downSize, "grey"))
        new_fig.add_trace(create_cloud_trace(temp_pcl, downSize, "red"))
        print("trace with rotation z added")
        return html.Div(dcc.Graph(id='rot_3d_scat', figure=new_fig), style={'height': '600', 'width':'2000', 'margin':'20px'})
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
        new_fig.data = []
        temp_pcl = copy.deepcopy(scan_pcd)
        if not z_bool:
            temp_pcl = rotate_Z(temp_pcl, z_value)
        if not x_bool:
            temp_pcl = rotate_X(temp_pcl, x_value)
        temp_pcl = rotate_Y(temp_pcl, rot_val[len(rot_val)-1])
        new_fig.add_trace(create_cloud_trace(pcd, downSize, "grey"))
        new_fig.add_trace(create_cloud_trace(temp_pcl, downSize, "red"))
        print("trace with rotation y added")
        return html.Div(dcc.Graph(id='rot_3d_scat', figure=new_fig), style={'height': '600', 'width':'2000', 'margin':'20px'})
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
        new_fig.data = []
        temp_pcl = copy.deepcopy(scan_pcd)
        if not y_bool:
            temp_pcl = rotate_Y(temp_pcl, y_value)
        if not z_bool:
            temp_pcl = rotate_Z(temp_pcl, z_value)
        temp_pcl = rotate_X(temp_pcl, rot_val[len(rot_val)-1])
        new_fig.add_trace(create_cloud_trace(pcd, downSize, "grey"))
        new_fig.add_trace(create_cloud_trace(temp_pcl, downSize, "red"))
        print("trace with rotation x added")
        return html.Div(dcc.Graph(id='rot_3d_scat', figure=new_fig), style={'height': '600', 'width':'2000', 'margin':'20px'})
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
            rotate_Y(scan_pcd, y_value)
        if not z_bool:
            rotate_Z(scan_pcd, z_value)
        if not x_bool:
            rotate_X(scan_pcd, x_value)
        # compute the threshold for ICP based on average of nearest neighbor distance
        density_target = o3d.geometry.PointCloud.compute_nearest_neighbor_distance(pcd)
        threshold = (np.average(density_target)) * 0.009
        # threshold = 0.05 default value from open3D
        print("Threshold:", threshold)
        # a transformation matrix with rotation = 0, translation = 0 and scale = 1
        trans_init = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0],[0.0, 0.0, 0.0, 1.0]]
        # compute transformation from ICP
        reg_p2p = o3d.registration.registration_icp(
            scan_pcd, pcd, threshold, trans_init,
            o3d.registration.TransformationEstimationPointToPoint())

        # apply transformation to scan
        scan_pcd.transform(reg_p2p.transformation)
        #o3d.visualization.draw_geometries([source_cent, target])

        new_fig.data = []
        # adds the trace of the scan and the model (both pcls)
        new_fig.add_trace(create_cloud_trace(pcd, downSize, "grey"))
        new_fig.add_trace(create_cloud_trace(scan_pcd, downSize, "red"))
        return html.Div(dcc.Graph(id='rot_3d_scat', figure=new_fig), style={'height': '600', 'width':'2000', 'margin':'20px'})

#callback to change the tab with a button
@app.callback(
    Output("tabs-example", "value"),
    [Input("button-yes", "n_clicks"), Input("button-not", "n_clicks")],
    )
def change_tab(clicks1, clicks2):
    ctx = dash.callback_context
    if ctx.triggered[0]['value'] is None:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if ctx.triggered:
        print(button_id)
        if button_id == "button-not":
            return "tab-2"
        elif button_id == "button-yes":
            return "tab-3"
    return "tab-1"

if __name__ == '__main__':
    app.run_server(debug=True)
