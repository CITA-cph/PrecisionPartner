import numpy as np
import copy
import open3d as o3d
from scipy.cluster.vq import kmeans
import math

# load the model as a point cloud
target = o3d.io.read_point_cloud("Ama_model_pc.ply")

# load the scan as a mesh and make it a point cloud with colors
scanMesh = o3d.io.read_triangle_mesh("ama_scan_pc_3points.ply")
source = o3d.geometry.PointCloud()
source.points = scanMesh.vertices
source.colors = scanMesh.vertex_colors

# load points on the model corresponding to trackers
trackers = o3d.io.read_point_cloud("trackers_points.ply")
# make numpy array and calculate the number of trackers
trackers_points = np.asarray(trackers.points)
trackersNum = np.size(trackers_points[:, 0])


def ColorFilter(pcl):
    col = np.asarray(pcl.colors)
    pts = np.asarray(pcl.points)

    # converts color values to 0-250
    col = col * 250

    # creates min and max of rgb values
    green = color_array = np.array([
        [0, 85, 0],  # dark green
        [90, 255, 73]])  # light green

    mask = np.all(
        np.logical_and(
            np.min(green, axis=0) < col,
            col < np.max(green, axis=0)
        ),
        axis=-1
    )
    # array with green points
    newpts = pts[mask]

    # find segment centroids
    raw_centroids = kmeans(newpts, trackersNum)
    centroids_f = raw_centroids[0]

    return centroids_f


def calculate_distances(points01):
    # dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    # distance between p1-p2
    d1 = math.sqrt((points01[0, 0] - points01[1, 0]) ** 2 + (points01[0, 1] - points01[1, 1]) ** 2)
    # distance between p2-p3
    d2 = math.sqrt((points01[2, 0] - points01[1, 0]) ** 2 + (points01[2, 1] - points01[1, 1]) ** 2)
    # distance between p3-p1
    d3 = math.sqrt((points01[2, 0] - points01[0, 0]) ** 2 + (points01[2, 1] - points01[0, 1]) ** 2)
    # make a distance array with the points corresponding to each distance
    distances = np.zeros((3, 7))
    distances[0, 0] = d1
    distances[1, 0] = d2
    distances[2, 0] = d3
    distances[0, 1:4] = points01[0, :]
    distances[0, 4:7] = points01[1, :]
    distances[1, 1:4] = points01[1, :]
    distances[1, 4:7] = points01[2, :]
    distances[2, 1:4] = points01[2, :]
    distances[2, 4:7] = points01[0, :]
    distances = np.round(distances, 1)
    distances = distances[np.argsort(distances[:, 0])]
    return distances


def match_points_trackers(num_pts):
    index_cp1 = np.where(centroids_corr == centroids_corr[num_pts, 0])
    index_cp1 = np.asarray(index_cp1)
    rows = trackers.shape[0]
    col = trackers.shape[1]
    pts = np.empty(())
    for i in range(0, rows):
        for j in range(0, col):
            index_p = np.where(trackers == trackers[i, j])
            index_p = np.asarray(index_p)
            # if the rows of the points match
            if np.array_equal(index_p[0, :], index_cp1[0, :]):
                pts = np.append(pts, trackers[i, j])
    # keeps only the first part of the sets of points
    pts = pts[1:4]
    return pts


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


if __name__ == "__main__":
    print("Visualization of two point clouds before manual alignment")
    draw_registration_result(source, target, np.identity(4))

    # find centroids of trackers on scan
    centroids_points = ColorFilter(source)
    print("Trackers' position on scan:", centroids_points)
    print("Trackers' position on model:", trackers_points)

    # find the distances between the points
    # returns a list with the first column containing the 3 distances
    # and 6 more columns containing 2 points in each row corresponding to the distance
    # the list is sorted based on the distance column
    dis_centroids = calculate_distances(centroids_points)
    dis_trackers = calculate_distances(trackers_points)

    # centroids_corr: array with the sets of points only (without distances)
    # in centroids_corr each point in each row corresponds to one of the two in the same row in trackers
    # we need to find which one of the two is the one we're looking for
    # centroids: p1 p2 or p2 p1 trackers: p1' p2' or p2' p1' <-- for each row
    centroids_corr = dis_centroids[:, 1:7]
    # trackers: array with the sets of points
    trackers = dis_trackers[:, 1:7]

    trackers_corr = np.empty((0, 3))
    # match_points_trackers search centroids_corr for each x value
    # and finds in which rows you find again the same value
    # then it searches trackers and check for each x value
    # if you can find it in the same rows as the one in centroids_corr
    # so if two points in both arrays exist in the same 2 rows, then they correspond
    trackers_corr = np.stack((np.asarray(match_points_trackers(0)),
                              np.asarray(match_points_trackers(1)),
                              np.asarray(match_points_trackers(2))),
                             axis=0)
    # keeps only the first part of the sets of points
    centroids_corr = centroids_corr[:, 0:3]
    print('correspondence set', trackers_corr, centroids_corr)

    centroids = o3d.geometry.PointCloud()
    centroids.points = o3d.utility.Vector3dVector(centroids_corr)

    trackers2 = o3d.geometry.PointCloud()
    trackers2.points = o3d.utility.Vector3dVector(trackers_corr)

    # numpy array with indices
    corr = np.array([(0., 0.), (1., 1.), (2., 2.)])

    # estimate rough transformation using correspondences
    # aligns the point cloud of the trackers and the centroids
    # use this transformation in ICP with the models
    # another way to do this: add points on the main clouds, but this is faster
    print("Compute a rough transform using the correspondences given by trackers' position")
    p2p = o3d.registration.TransformationEstimationPointToPoint()
    trans_init = p2p.compute_transformation(centroids, trackers2,
                                            o3d.utility.Vector2iVector(corr))
    draw_registration_result(source, target, trans_init)

    # compute the threshold based on average of nearest neighbor distance
    density_target = o3d.geometry.PointCloud.compute_nearest_neighbor_distance(target)
    print("Density in target:", density_target)
    threshold = (np.average(density_target)) * 0.009
    # threshold = 0.05 default value from open3D
    print("Threshold:", threshold)

    # point-to-point ICP for refinement
    print("Perform point-to-point ICP refinement")
    reg_p2p = o3d.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.registration.TransformationEstimationPointToPoint())
    draw_registration_result(source, target, reg_p2p.transformation)
