import open3d as o3d
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def main():
    # File paths
    stl_file = "office.stl"
    ply_file = "office.ply"
    pointcloud_file = "office_pointcloud.ply"

    try:
        # Step 1: Read the STL file
        logging.info(f"Reading STL file: {stl_file}")
        mesh = o3d.io.read_triangle_mesh(stl_file)
        if mesh.is_empty():
            logging.error("Failed to load the STL file. The mesh is empty.")
            return
        logging.info("Successfully loaded STL file.")

        # Step 2: Sample points from the mesh to create a point cloud
        logging.info("Sampling points to create a point cloud...")
        pointcloud = mesh.sample_points_poisson_disk(100000)
        logging.info("Point cloud sampling completed.")

        # Step 3: Save the STL mesh as a PLY file
        logging.info(f"Saving mesh to PLY file: {ply_file}")
        o3d.io.write_triangle_mesh(ply_file, mesh)
        logging.info("Mesh saved successfully.")

        # Step 4: Save the point cloud as a PLY file
        logging.info(f"Saving point cloud to PLY file: {pointcloud_file}")
        o3d.io.write_point_cloud(pointcloud_file, pointcloud)
        logging.info("Point cloud saved successfully.")

        # Step 5: Visualize the mesh
        logging.info("Visualizing the mesh...")
        o3d.visualization.draw_geometries([mesh], window_name="Mesh Visualization")
        logging.info("Mesh visualization completed.")

        # Step 6: Visualize the point cloud
        logging.info("Visualizing the point cloud...")
        o3d.visualization.draw_geometries([pointcloud], window_name="Point Cloud Visualization")
        logging.info("Point cloud visualization completed.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
