"""
In its simplest definition, point density describes the number of points in a given area.
Commonly the point density is given for one square meter and therefore uses the unit pts/m².
Point spacing on the other hand is defined as the distance between two adjacent points.

CLOUD COMPARE: Two methods can be used to compute the density:
either 'Precise': the density is estimated by counting for each point the number of neighbors N
or 'Approximate': the density is simply estimated by determining the distance to the nearest neighbor
(which is generally much faster). This distance is considered as being equivalent to the above spherical
neighborhood radius R (and N = 1).
"""
# Generating points on model, num of pts on model same as scan
# downsampling model's pcl with a voxel size = density of the scan
# Given: the scan will always be equal or less data from the model

import numpy as np
import open3d as o3d

print("--Reading model as a mesh file--")
Model_mesh = o3d.io.read_triangle_mesh \
    ("Q:/CITA/DIGITAL_FORMATIONS/"
     "2019-precisionPartners innobyg/03_Work/09_Python/alignment_registration/3D models/"
     "Amabrogade_model.ply")

print("--Reading scan as a mesh file--")
Scan_mesh = o3d.io.read_triangle_mesh("Q:/CITA/DIGITAL_FORMATIONS/"
                                      "2019-precisionPartners innobyg/03_Work/09_Python/alignment_registration/3D models/"
                                      "balcony_clean.ply")
print("--Creating point cloud from scan's vertices--")
scan = o3d.geometry.PointCloud()
scan.points = Scan_mesh.vertices
scan.colors = Scan_mesh.vertex_colors


def density(pcl):
    density_pcl = o3d.geometry.PointCloud.compute_nearest_neighbor_distance(pcl)
    density_av = np.average(density_pcl)
    return density_av


num_pts_scan = np.asarray(scan.points)
print("Number of pts in scan: ", np.size(num_pts_scan, 0))

# multiplied by 10, 10 times scan - model
model = Model_mesh.sample_points_uniformly(10*(np.size(num_pts_scan, 0)))

density_scan = density(scan)
model_down = model.voxel_down_sample(density_scan)

# check
density_model_down = density(model_down)
#density_dif = abs(density_scan-density_model)
#print("density difference", density_dif)
print("density scan", density_scan)
print("density model", density_model_down)


o3d.visualization.draw_geometries([model_down, scan])


