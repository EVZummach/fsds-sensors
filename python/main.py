import os
import cv2
import sys
import time
import utils
import numpy as np

fsds_lib_path = os.path.join(os.path.expanduser("~"), "Formula-Student-Driverless-Simulator", "python")
sys.path.insert(0, fsds_lib_path)
import fsds

client = fsds.FSDSClient()
client.confirmConnection()
client.reset()

# Some devices allow you to define multiple instances. Simply adjust the settings.json 
# located at the FSDS executable adding a new sensor in the specified area and configs
# and to call it, simply define its name below

cam = utils.Camera(client=client, save_path='camera_data', camera_name='cam1')
# cam2 = utils.Camera(client=client, save_path='camera_data', camera_name='cam2')
lidar = utils.LiDAR(client=client, save_path='lidar_data', lidar_name='Lidar1')
lidar.init_visualizer()
imu = utils.IMU(client=client, save_path='imu_data', imu_name='Imu')
gps = utils.GPS(client=client, save_path='gps_data', gps_name='Gps')
car = utils.CarState(client=client, save_path='car_data', car_name='FSCar')

all_data = []

while True:

    # Get Camera data
    img = cam()
    cv2.imshow(f'{cam.name} Image', img)
    cv2.waitKey(1)

    # Get LiDAR data
    lidar_data = lidar()

    # Get IMU data
    imu_data = imu()
    #timestamp
    print(f'timestamp: {imu_data["timestamp"]}') 
    #w, x, y, z orientation values
    print(f'orientation: {imu_data["orientation"]}')
    #x, y, z velocity values
    print(f'velocity: {imu_data["velocity"]}')
    #x, y, z acceleration values
    print(f'acceleration: {imu_data["acceleration"]}')
    #roll, pitch, yaw angles (based on orientation)
    print(f'calculated angles: {imu_data["angles"]}')

    # Get GPS data
    gps_data = gps()
    #timestamp
    print(f'timestamp: {gps_data["timestamp"]}') 
    #timestamp-utc
    print(f'timestamp utc: {gps_data["timestamp_utc"]}') 
    #standard deviation of horizontal position error
    print(f'eph: {gps_data["timestamp"]}') 
    #standard deviation of vertical position error
    print(f'epv: {gps_data["timestamp"]}') 
    #altitude, latitude, longitude
    print(f'geopoint: {gps_data["timestamp"]}') 
    #x, y, z velocities
    print(f'velocity: {gps_data["timestamp"]}') 

    # Get ground truth data
    car_data = car()
    #timestamp
    print(f'timestamp: {car_data["timestamp"]}') 
    #w, x, y, z orientation values
    print(f'orientation: {car_data["orientation"]}')
    #x, y, z velocity values
    print(f'velocity: {car_data["velocity"]}')
    #x, y, z acceleration values
    print(f'acceleration: {car_data["acceleration"]}')
    #roll, pitch, yaw angles (based on orientation)
    print(f'calculated angles: {car_data["angles"]}')

    # Example to plot LiDAR data based on car state
    corrected, oriented = lidar.calculate(car_data['orientation'], car_data['position'])
    lidar.visualizer(oriented)

client.reset()