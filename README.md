# Installation
This library was developed to help interface with the Formula-Student-Driverless-Simulator and its sensors. To use it, simply clone this repo in your file, and install the python libraries needed.

To simplify the usage, create a venv and install the project library. To create a venv, simply type the following command with the shell on the clone path <br>
`python -m venv .` <br>
`pip install -r python\requirements.txt`

## Notes

Don't forget to clone the FSDS repo, and download the FSDS executable file!


# Usage

You should add your modifications in the main.py file and run it with the simulator open and on a track. An example of getting the devices' data is already in the main loop, simply adjust according to your needs.

Some devices allow you to define multiple instances. Simply adjust the settings.json located at the FSDS executable adding a new sensor in the specified area and configs and to call it, simply define its name as specified in the config file. As an example: the settings.json has two cameras named 'camFW' (FW as in FrontWing) and 'camMH' (MH as in MainHoop),respectively. To get both their images, simply define them before the main loop:

```
cam_fw = utils.Camera(client=client, save_path='camera_data', camera_name='camFW')
cam_mh = utils.Camera(client=client, save_path='camera_data', camera_name='camMH')
```

You can access their images in the main loop by calling

```
img_fw = cam_fw()
img_mh = cam_mh()
```

This will return a numpy array with the RGB format for each image.

To stop the code execution, simply `Ctrl+C` in your terminal running the script.

# Common sensor positions:

This will be updated, but the main ones are listed below:

    Main Hoop:
        "X":-0.5,
        "Y":0.0,
        "Z":1,

    Front Hoop:
        "X":1.0,
        "Y":0.0,
        "Z":0.3,

    Front Wing:
        "X":0.75,
        "Y":0.0,
        "Z":0.65,