import os
import cv2
import sys
import glob
import math
import numpy as np
import pyvista as pv
import open3d as o3d
import matplotlib.pyplot as plt

fsds_lib_path = os.path.join(os.path.expanduser("~"), "Formula-Student-Driverless-Simulator", "python")
sys.path.insert(0, fsds_lib_path)

import fsds

def create_dest_folder(dest_path, extension):
    """
    Create a folder on a specified path, along with the sensor name
    :dest_path: output path (ex: lidar_data, car_data, imu_data)
    :extension: sensor name (ex: Lidar1, cam1, Imu)
    """
    index = len(glob.glob(f'{dest_path}/run*'))        
    path = f'{dest_path}/run_{index}/{extension}'
    os.makedirs(path, exist_ok=True)
    return path

def create_dest_file(save_path, data):
    """
    Create a log file of the sensor data. It will automatically detect the file index based on the output path
    :save_path: complete relative output path (ex: lidar_data/run_0/)
    :data: sensor data in numpy array format
    """
    index = len(os.listdir(save_path))
    file_path = os.path.join(save_path, f'{index}.txt')
    np.savetxt(file_path, data)
    return file_path

class Camera():
    def __init__(self, client, save_path, camera_name, save=False):
        """
        Class to store camera video on specified folder.

        :client: FSDS Client
        :path: Path to save Camera data. It will save as save_path/run_x/camera_name
        :camera_name: Camera name according to the settings.json file
        """
        self.save = save
        self.path = create_dest_folder(save_path, camera_name) if save else save_path

        self.name = camera_name
        self.client = client

        images = client.simGetImages([fsds.ImageRequest(camera_name=camera_name, image_type=fsds.ImageType.Scene, pixels_as_float = False, compress=False)])
        image = images[0]

        if self.save:
            self.writer = cv2.VideoWriter(f'{self.path}/simulator-record.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (image.height, image.width))

    def __call__(self):
        images = self.client.simGetImages([fsds.ImageRequest(camera_name=self.name, image_type=fsds.ImageType.Scene, pixels_as_float = False, compress=False)])
        image = images[0]
        img1d = np.fromstring(image.image_data_uint8, dtype=np.uint8) 
        img_rgb = img1d.reshape(image.height, image.width, 3)
        if self.save:
            self.writer.write(img_rgb)
        return img_rgb

    def end(self):
        if self.save:
            self.writer.release()


# Faltando:
#   Montar um pipeline de filtro do chão
#   Pipeline de visualização dos dados
class LiDAR():
    def __init__(self, client, save_path, lidar_name, save=False):
        """
        Class to store LiDAR data on specified folder.

        :client: FSDS Client
        :save_path: Path to save LiDAR data. It will save as save_path/run_x/lidar_name
        :lidar_name: LiDAR name according to the settings.json file
        """
        self.save = save
        self.path = create_dest_folder(save_path, lidar_name) if save else save_path
        self.client = client
        self.name = lidar_name
        self.points = []

    def __call__(self):
        lidardata = self.client.getLidarData(lidar_name=self.name)
        self.lidardata = lidardata
        points = np.array(lidardata.point_cloud, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0]/3), 3))
        self.points = points
        #self.points.append(points)
        if self.save:
            create_dest_file(self.path, points)
        return points

    def init_visualizer(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(height=640, width=640)
        self.pcd = o3d.geometry.PointCloud()
        points = self.__call__()
        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.vis.add_geometry(self.pcd)


    def visualizer(self, input_points):
        # print("vis")
        points = self.corrected_points if input_points is None else input_points
        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.vis.poll_events()
        self.vis.update_geometry(self.pcd)
        self.vis.update_renderer()
        

    # oriented is the angle adjusted local LiDAR data. It will adjust the LiDAR data based on
    # the roll, pitch and yaw angles
    # corrected is the angle adjusted and offseted LiDAR data. Example: car is in position (0, 0)
    # and a new reading is done at positions (5, 10). corrected will be a global LiDAR reading with 
    # an offset of 5 and 10, respectivelly (while adjusting based on angles)
    def calculate(self, orientation, position):
        q0, q1, q2, q3 = orientation
        rotation_matrix = np.array(([1-2*(q2*q2+q3*q3),2*(q1*q2-q3*q0),2*(q1*q3+q2*q0)],
                                    [2*(q1*q2+q3*q0),1-2*(q1*q1+q3*q3),2*(q2*q3-q1*q0)],
                                    [2*(q1*q3-q2*q0),2*(q2*q3+q1*q0),1-2*(q1*q1+q2*q2)]))
        
        final_x, final_y, final_z = [], [], []
        oriented_x, oriented_y, oriented_z = [], [] ,[]
        
        for i in range(0, len(self.lidardata.point_cloud), 3):
            xyz = self.lidardata.point_cloud[i:i+3]
            corrected_x, corrected_y, corrected_z = np.matmul(rotation_matrix, np.asarray(xyz))
            
            x = corrected_x + position[0]
            y = corrected_y + position[1]
            z = corrected_z + position[2]

            oriented_x.append(corrected_x)
            oriented_y.append(corrected_y)
            oriented_z.append(corrected_z)

            final_x.append(x)
            final_y.append(y)
            final_z.append(z)
        
        self.corrected_data = np.array([final_x, final_y, final_z]).T
        self.oriented_data = np.array([oriented_x, oriented_y, oriented_z]).T
        #create_dest_file(self.path, data)
        return self.corrected_data, self.oriented_data
    
    def filter(self, points):
        points = points[points[:, 2] >= -0.04]
        return points

class IMU():
    def __init__(self, client, save_path, imu_name, save=False):
        self.path = create_dest_folder(save_path, imu_name) if save else save_path
        self.save = save
        self.client = client
        self.name = imu_name
    
    def __call__(self):
        imu_data = self.client.getImuData(imu_name = self.name)
        self.timestamp = imu_data.time_stamp
        self.orientation = [imu_data.orientation.w_val, imu_data.orientation.x_val, imu_data.orientation.y_val, imu_data.orientation.z_val]
        self.velocity = [imu_data.angular_velocity.x_val, imu_data.angular_velocity.y_val, imu_data.angular_velocity.z_val]
        self.acceleration = [imu_data.linear_acceleration.x_val, imu_data.linear_acceleration.y_val, imu_data.linear_acceleration.z_val]
        self.angles = self.calculate_angles()

        self.data = {
            'timestamp':self.timestamp,
            'orientation':self.orientation,
            'velocity':self.velocity,
            'acceleration':self.acceleration,
            'angles':self.angles
            }
        
        if self.save:
            create_dest_file(self.path, self.angles)
        
        return self.data

    def calculate_angles(self):
        w = self.orientation[0]
        x = self.orientation[1]
        y = self.orientation[2]
        z = self.orientation[3]

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)
     
        return roll, pitch, yaw # in radians

class GPS():
    def __init__(self, client, save_path, gps_name, save=False):
        self.path = create_dest_folder(save_path, gps_name) if save else save_path     
        self.client = client
        self.name = gps_name

    def __call__(self):
        gps_data = self.client.getGpsData(gps_name = self.name)
        self.timestamp = gps_data.time_stamp
        self.timestamp_utc = gps_data.gnss.time_utc
        self.eph = gps_data.gnss.eph
        self.epv = gps_data.gnss.epv
        self.geo_point = [gps_data.gnss.geo_point.altitude, gps_data.gnss.geo_point.latitude, gps_data.gnss.geo_point.longitude]
        self.velocity = [gps_data.gnss.velocity.x_val, gps_data.gnss.velocity.y_val, gps_data.gnss.velocity.z_val]

        self.data = {
            'timestamp':self.timestamp,
            'timestamp-utc':self.timestamp_utc,
            'eph':self.eph,
            'epv':self.epv,
            'geo-point':self.geo_point,
            'velocity':self.velocity
            }
        
        return self.data

class CarState():
    def __init__(self, client, save_path, car_name, save=False):
        self.save = save
        self.path = create_dest_folder(save_path, car_name) if save else save_path
        self.client = client
        self.name = car_name

    def __call__(self):
        kinematic_data = self.client.simGetGroundTruthKinematics()
        self.position = [kinematic_data.position.x_val, kinematic_data.position.y_val, kinematic_data.position.z_val]
        self.orientation = [kinematic_data.orientation.w_val, kinematic_data.orientation.x_val, kinematic_data.orientation.y_val, kinematic_data.orientation.z_val]
        self.angles = self.calculate_angles()

        self.data = {
            'position':self.position,
            'orientation':self.orientation,
            'angles':self.angles
            }
        
        if self.save:
            create_dest_file(self.path, np.hstack([self.position, self.angles]))

        return self.data

    def calculate_angles(self):
        w = self.orientation[0]
        x = self.orientation[1]
        y = self.orientation[2]
        z = self.orientation[3]

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)
     
        return roll, pitch, yaw # in radians
    
class Visualization():
    def __init__(self):
        print("Making some tests")