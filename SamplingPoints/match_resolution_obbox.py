# Sample points on model based on scan's density
# MODEL IS ALIGNED WITH THE AXIS
# Open3d's oriented bounding box doesn't compute the exact minimum bbox
# instead  -> finding axis aligned bbox with min volume while rotating pcl, but more expensive
# see match_resolution_minbbox.py
# understand the error = in this case they should have approximately the same number of points

import numpy as np
import open3d as o3d

print("--Reading model as a mesh file--")
Model_mesh = o3d.io.read_triangle_mesh\
    ("Q:/CITA/DIGITAL_FORMATIONS/"
     "2019-precisionPartners innobyg/03_Work/09_Python/alignment_registration/3D models/"
     "Amabrogade_model.ply")

print("--Reading scan as a point cloud file--")
Scan_mesh = o3d.io.read_triangle_mesh("Q:/CITA/DIGITAL_FORMATIONS/"
                                 "2019-precisionPartners innobyg/03_Work/09_Python/alignment_registration/3D models/"
                                 "balcony_clean.ply")
scan = o3d.geometry.PointCloud()
scan.points = Scan_mesh.vertices
scan.colors = Scan_mesh.vertex_colors

def density(pcl):
    density_pcl = o3d.geometry.PointCloud.compute_nearest_neighbor_distance(pcl)
    density_av = np.average(density_pcl)
    return density_av

# find the bbox volumes to compare
bbox_scan = scan.get_oriented_bounding_box()
bbox_model = Model_mesh.get_oriented_bounding_box()

# visualise
bbox_scan_ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(bbox_scan)
bbox_model_ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(bbox_model)
o3d.visualization.draw_geometries([scan, bbox_scan_ls, Model_mesh, bbox_model_ls])

num_pts_scan = np.asarray(scan.points)
print("--Number of pts in scan: ", np.size(num_pts_scan, 0))
num_pts_model = (o3d.geometry.OrientedBoundingBox.volume(bbox_model) * np.size(num_pts_scan, 0)) \
                / o3d.geometry.OrientedBoundingBox.volume(bbox_scan)
print("--Number of pts in model: ", int (num_pts_model))

model = Model_mesh.sample_points_uniformly(int (num_pts_model))

# check
density_scan = density(scan)
density_model = density(model)
print("density scan", density_scan)
print("density model", density_model)

o3d.visualization.draw_geometries([scan, model])



