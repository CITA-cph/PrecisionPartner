# Sample points on model based on scan's density
# MODEL IS ALIGNED WITH THE AXIS
# Open3d's oriented bounding box doesn't compute the exact minimum bbox
# instead  -> finding axis aligned bbox with min volume while rotating pcl, but more expensive
# understand the error = in this case they should have approximately the same number of points
# FINDINGS_BEST

import numpy as np
import open3d as o3d
import copy
from math import cos, sin

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


def get_bbox_from_rotation(pcl, matrix):
    pcl_temp = copy.deepcopy(pcl)
    o3d.geometry.PointCloud.rotate(pcl_temp, matrix)
    bbox = pcl_temp.get_axis_aligned_bounding_box()
    return bbox


def min_bbox_rotation(pcl):
    """
    http://web.cs.iastate.edu/~cs577/handouts/rotation.pdf
    Rotzyx(φ, θ, ψ) = Rotz(φ) · Roty(θ) · Rotx(ψ) =
    ([cos φ − sin φ 0], [sin φ cos φ 0], [0 0 1])
    ([cos θ 0 sin θ], [0 1 0], [− sin θ 0 cos θ])
    ([1 0 0], [0 cos ψ − sin ψ], [0 sin ψ cos ψ])
    """
    step_degrees = 1
    q = 1
    j = 1
    i = 1

    print("--Loop into rotations based on x--")
    # FOR X
    bbox_best_x_volume = 90000000000000000.
    while q <= 360:
        rot_matrix = np.asarray([[1, 0, 0],
                                 [0, cos(q / 57.2958), -sin(q / 57.2958)],
                                 [0, sin(q / 57.2958), cos(q / 57.2958)]])
        BboxX = get_bbox_from_rotation(pcl, rot_matrix)
        if o3d.geometry.AxisAlignedBoundingBox.volume(BboxX) < bbox_best_x_volume:
            bbox_best_x_volume = o3d.geometry.AxisAlignedBoundingBox.volume(BboxX)
            rot_matrix_bestX = rot_matrix
        q += step_degrees

    o3d.geometry.PointCloud.rotate(pcl, rot_matrix_bestX)

    print("--Loop into rotations based on y--")
    # FOR Y
    bbox_best_y_volume = 90000000000000000.
    while j <= 360:
        rot_matrix = np.asarray([[cos(j / 57.2958), 0, sin(j / 57.2958)],
                                 [0, 1, 0],
                                 [-sin(j / 57.2958), 0, cos(j / 57.2958)]])
        BboxY = get_bbox_from_rotation(pcl, rot_matrix)
        if o3d.geometry.AxisAlignedBoundingBox.volume(BboxY) < bbox_best_y_volume:
            bbox_best_y_volume = o3d.geometry.AxisAlignedBoundingBox.volume(BboxY)
            rot_matrix_bestY = rot_matrix
        j += step_degrees

    o3d.geometry.PointCloud.rotate(pcl, rot_matrix_bestY)

    print("--Loop into rotations based on z--")
    # FOR Z
    bbox_best_z_volume = 90000000000000000.
    while i <= 360:
        rot_matrix = np.asarray([[cos(i / 57.2958), -sin(i / 57.2958), 0],
                                 [sin(i / 57.2958), cos(i / 57.2958), 0],
                                 [0, 0, 1]])
        BboxZ = get_bbox_from_rotation(pcl, rot_matrix)
        if o3d.geometry.AxisAlignedBoundingBox.volume(BboxZ) < bbox_best_z_volume:
            bbox_best_z_volume = o3d.geometry.AxisAlignedBoundingBox.volume(BboxZ)
            rot_matrix_bestZ = rot_matrix
        i += step_degrees

    o3d.geometry.PointCloud.rotate(pcl, rot_matrix_bestZ)

    return pcl


def visualise_bbox(pcl, bbox):
    bbox_ls = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox)
    return o3d.visualization.draw_geometries([pcl, bbox_ls])


def density(pcl):
    density_pcl = o3d.geometry.PointCloud.compute_nearest_neighbor_distance(pcl)
    density_av = np.average(density_pcl)
    return density_av


# find the bbox volumes to compare
# rotate scan until you find the abbox with the min volume
min_scan = min_bbox_rotation(scan)
bbox_scan = min_scan.get_axis_aligned_bounding_box()

bbox_model = Model_mesh.get_axis_aligned_bounding_box()

visualise_bbox(min_scan, bbox_scan)

num_pts_scan = np.asarray(scan.points)
print("--Number of pts in scan: ", np.size(num_pts_scan, 0))
num_pts_model = (o3d.geometry.AxisAlignedBoundingBox.volume(bbox_model) * np.size(num_pts_scan, 0)) \
                / o3d.geometry.AxisAlignedBoundingBox.volume(bbox_scan)
print("--Number of pts in model: ", int(num_pts_model))

model = Model_mesh.sample_points_uniformly(int(num_pts_model))

# check
density_scan = density(scan)
density_model = density(model)
print("density scan", density_scan)
print("density model", density_model)

o3d.visualization.draw_geometries([scan, model])
