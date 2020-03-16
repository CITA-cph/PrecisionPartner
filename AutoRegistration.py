# examples/Python/Advanced/global_registration.py

import open3d as o3d
import numpy as np
import copy
import random
import math


def draw_registration_result(source, target, transformation):
    # transform and paint create permanent changes to the pcls
    # creating a copy to visualize
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def preprocess_point_cloud(pcd, voxel_size):
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


def prepare_dataset():
    # import the model as a point cloud
    target = o3d.io.read_point_cloud("Ama_model_pc.ply")

    # import the scan as a mesh and create a point cloud with its vertices
    mesh = o3d.io.read_triangle_mesh("01_balcony_mold_03.ply")
    source = o3d.geometry.PointCloud()
    source.points = mesh.vertices

    # find the voxel size based on average nearest neighbor distance of the scan
    density_source = o3d.geometry.PointCloud.compute_nearest_neighbor_distance(source)
    print("Density", np.average(density_source))
    voxel_size = 67 * (np.average(density_source))
    voxel_size = np.around(voxel_size, decimals=0)
    print('Voxel Size:', voxel_size)

    # random transformation matrix
    test = random.randint(15, 45) * 0.0174533
    trans_init = np.asarray([[math.cos(test), math.sin(test), 0.0, 0.0], [-math.sin(test), math.cos(test), 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    return source, target, source_down, target_down, source_fpfh, target_fpfh, voxel_size


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print("   RANSAC registration on downsampled point clouds.")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))
    return result


def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    print("   Refine registration")
    print("   Point-to-plane ICP registration with distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.registration.TransformationEstimationPointToPoint())
    return result


if __name__ == "__main__":
    source, target, source_down, target_down, source_fpfh, target_fpfh, voxel_size = \
        prepare_dataset()

    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)

    draw_registration_result(source_down, target_down, result_ransac.transformation)

    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    result_icp = refine_registration(source, target, source_fpfh, target_fpfh, voxel_size)

    draw_registration_result(source, target, result_icp.transformation)
