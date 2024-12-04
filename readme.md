# Python: Augmented Reality

This repository is intended to help you get started with the topic of **Python: Augmented Reality**. Following topics are explained and have examples:

- Camera calibration
- ArUco marker generation
- ArUco marker detection

## Minimum requirements

- Python 3.7.x installed
- USB camera

## Project structure

```shell
# clone the project to local
$ git clone https://github.com/Lupin3000/AugmentedReality.git

# change into cloned directory
$ cd AugmentedReality/
```

The final folders and file structure of the project (_if no calibration has yet been carried out and no markers have been generated_).

```shell
# list all files/folder (optional)
$ tree .
|____.gitignore
|____requirements.txt
|____ar_videos.py
|____ar_marker_ids.py
|____ar_images.py
|____dev
| |____show_calbraion.py
| |____img
| | |____pattern.png
| |____get_calibration.py
| |____generate_marker.py
|____src
| |____videos
| | |____video_1.mp4
| | |____video_0.mp4
| |____photos
| | |____monk_1.jpg
| | |____monk_0.jpg
```

**Description**

The root folder of the project contains the files: `requirements.txt`, `ar_marker_id.py`, `ar_images.py` and `ar_videos.py`. Except for the `requirements.txt` file, which is used to install the necessary Python modules/libraries, all other Python files serve as examples for Augmented Reality (AR).

The `dev/` folder contains Python scripts that support you, for example, with camera calibration and ArUco marker generation. In the src folder you will find images and videos that are used for the AR examples.

In the `dev/img/` subfolder you will find the file `pattern.png`. This is needed to be printed out for camera calibration.

When you create new markers using the Python script `dev/generate_marker.py`, they are saved to a new subfolder within the dev folder.

If you have carried out a camera calibration, you will find the file `camera_params.npz` in the `src/` folder. This file will be loaded in the AR examples (_if available_).

## Prepare a local development environment
Various Python modules/libraries are used in this project. It is therefore recommended to use Python Virtual Environment. The necessary modules/libraries are listed in the requirements.txt file.

The next commands show how the virtual environment is created and the installation is carried out.

```shell
# create virtual environment
$ python3 -m venv .venv

# list directory (optional)
$ ls -la

# activate virtual environment
$ .venv/bin/activate

# update pip (optional)
(.venv) $ pip3 install -U pip

# show content of requirements.txt (optional)
(.venv) $ cat requirements.txt

# install all modules/packages
(.venv) $ pip3 install -r requirements.txt

# list all modules/packages (optional)
(.venv) $ pip3 freeze
```

## Prepare and carry out calibration

Every camera has certain optical distortions, and it is sometimes difficult to get the necessary camera data from the manufacturers. Therefore, it is important to carry out the camera calibration! This is an essential step when using OpenCV ArUco markers as it ensures that the 2D image coordinates can be correctly converted into real 3D coordinates.

Calibration is typically done with a checkerboard pattern, which you can find in PNG format [here](dev/img/pattern.png) in the project.

> If you do not perform the calibrations, an imaginary value will be used in the AR examples.

**Calibration process**

1. Print out the `dev/img/pattern.png` file on A4 paper and glue the printed paper onto cardboard (_for stabilization_).
2. After printing, measure the length or width of a single cube! Depending on the printer, this can vary slightly. Then convert the value into the unit of measurement meters (_for example: 2.4cm is 0.024m_). Enter the value for the constant **SQUARE_SITE** in the Python script `dev/get_calibration.py`.
3. Provide good lighting for the area. Avoid strong shadows between the printed pattern and the camera. Also, avoid any light reflections on camera. 
4. Start the Python script `dev/get_calibration.py` and hold the pattern in front of the camera so that it is completely visible.
5. If you see artificial colored lines on the screen, press the **s-key** to perform the calibration and save the values.
6. To end the calibration proces and to stop the Python script, press the **q-key**.

> Each time you press the s-key, the calibration is carried out again and the values are overwritten in the file `src/camera_params.npz`. 
> 
> To display the values at any time later, you can create the Python script `dev/show_calibration.py`.

```shell
# run camera calibration
(.venv) $ python3 dev/get_calibration.py

# show camera values
(.venv) $ python3 dev/show_calibration.py
```

## Generate ArUco markers

You can create your own ArUco markers with the Python script `dev/generate_marker.py`. However, you must first adapt the constants **ARUCO_DICT_ID** as well as **ARUCO_MARKER_ID** and **ARUCO_MARKER_SIZE** to your needs.

- The respective ArUco Markers is set in the constant: **ARUCO_DICT_ID**.
- The respective ArUco Marker ID is set in the constant: **ARUCO_MARKER_ID**
- The size of ArUco Markers is set in in the constant: **ARUCO_MARKER_SIZE**

For example, the default of **ARUCO_DICT_ID** is set used is: `DICT_4X4_50`, which contains 50 predefined markers. The default value for **ARUCO_MARKER_ID** is `0`. You can change the value depending on the marker set you choose. For current default from `0` to `49`. The optimal value for **ARUCO_MARKER_SIZE** should be between `50` and `200`. Markers that are too small are harder to recognize.

To generate a marker, simply run the script `dev/generate_marker.py`.

```shell
# run marker generation
(.venv) $ python3 dev/generate_marker.py

# show created markers (optional)
(.venv) $ ls -la dev/markers/
```

Print out the marker(s) on paper, cut them and glue the printed paper onto cardboard (_for stabilization_).

> In the examples you still have to specify the length or height of the markers in meters in the constant: **MARKER_SIZE**.
> 
> So measure one of the created markers and change the value in `ar_images.py` and `ar_videos.py` if necessary.

## Run examples

- The file `ar_marker_ids.py` shows all detected markers and the respective ID.
- The file `ar_images.py` shows scaled pictures on marker position.
- The file `ar_videos.py` shows scaled video loops on marker position.

```shell
# run marker id detection
(.venv) $ python3 ar_marker_ids.py

# run marker image replacement
(.venv) $ python3 ar_images.py

# run marker video replacement
(.venv) $ python3 ar_videos.py
```
