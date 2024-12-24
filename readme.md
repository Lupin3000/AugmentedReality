# Python OpenCV: Augmented Reality

This repository is intended to introduce the topic of **Python: Augmented Reality** by ArUco marker detection.

Following details are explained and have examples:

- Camera calibration
- ArUco marker generation
- ArUco marker detection
- ArUco marker pose estimation
- Marker interactions

## Minimum requirements

[![Static](https://img.shields.io/badge/python->=3.7.x-green)](https://python.org)
[![Static](https://img.shields.io/badge/setuptolls-75.6-green)](https://setuptools.pypa.io/en/latest/)
[![Static](https://img.shields.io/badge/wheel-0.45-green)](https://wheel.readthedocs.io/en/stable/)
[![Static](https://img.shields.io/badge/OpenCV-4.10-green)](https://opencv.org)
[![Static](https://img.shields.io/badge/NumPy-2.2-green)](https://numpy.org)

- tested with Python 3.7.3 on UNIHIKER (_Debian_)
- tested with Python 3.12 on macOS
- USB or Laptop camera

## Project setup

**Clone Git repository**

```shell
# clone the project to local
$ git clone https://github.com/Lupin3000/AugmentedReality.git

# change into cloned directory
$ cd AugmentedReality/
```

## Project structure

The final folders and file structure of the project (_if no calibration has yet been carried out and no markers have been generated_).

**List all cloned local files and folders**

```shell
# list all files/folder (optional)
$ tree .
|____.gitignore
|____requirements.txt
|____example_01.py
|____example_02.py
|____example_03.py
|____example_04.py
|____example_05.py
|____example_06.py
|____example_07.py
|____example_08.py
|____example_09.py
|____example_10.py
|____example_11.py
|____example_12.py
|____example_13.py
|____dev
| |____img
| | |____pattern.png
| |____get_calibration.py
| |____show_calbraion.py
| |____generate_marker.py
|____src
| |____videos
| | |____[example files ...]
| |____photos
| | |____[example files ...]
```

**Project description**

| Directory  | Description                                                                                                |
|------------|------------------------------------------------------------------------------------------------------------|
| `root`     | contains the files: `.gitignore`, `requirements.txt` and `example_*.py`                                    |
| `dev/`     | contains Python scripts that support you, for example, with camera calibration and ArUco marker generation |
| `dev/img/` | you will find the file `pattern.png`. This pattern is needed for camera calibration                        |
| `src/`     | you will find images `src/photos/` and videos `src/videos/` that are used for the AR examples              |


> When you create new markers using the Python script `dev/generate_marker.py`, the markers are saved as `*.jpg` into the new subfolder within the `dev/markers/`.
> 
> If you have carried out a camera calibration, you will find the file `camera_params.npz` in the `src/` folder. This file will be loaded in the AR examples (_if available_).

## Prepare a local development environment

Various Python modules/libraries are used in this project. It is therefore recommended to use Python virtual environment. The necessary modules/libraries are listed in the `requirements.txt` file.

The next commands show how the Python virtual environment is created and the installation of required modules/libraries is carried out.

**Create/start virtualenv and install packages/modules:**

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

**Start and stop virtualenv**

If you want to interrupt the work on this project for some time, that's absolute not a problem! You can start and stop the Python virtual environment at any time.

```shell
# start virtualenv (project folder)
$ .venv/bin/activate

# stop virtualenv (any folder)
(.venv) $ deactivate
```

## Prepare and carry out calibration

Every camera has certain optical distortions, and it is sometimes difficult to get the necessary camera data from the manufacturers. Therefore, it is important to carry out the camera calibration! This is an essential step when using OpenCV ArUco markers as it ensures that the 2D image coordinates can be correctly converted into real 3D coordinates.

Calibration is typically done with a checkerboard pattern, which you can find in PNG format [here](dev/img/pattern.png) in the project.

> If you do not perform the calibrations, an imaginary value will be used in the AR examples.

**Calibration process**

1. Print out the `dev/img/pattern.png` file on A4 paper and glue the printed paper onto cardboard (_for stabilization_).
2. After printing, measure the length or width of a single cube! Depending on the printer, this can vary slightly. Then convert the value into the unit of measurement meters (_for example: 2.4cm is 0.024m_). Enter the value for the constant **SQUARE_SIZE** in the Python script `dev/get_calibration.py`.
3. Provide good lighting for the area. Avoid strong shadows between the printed pattern and the camera. Also, avoid any light reflections on camera. 
4. Start the Python script `dev/get_calibration.py` and hold the pattern in front of the camera so that it is completely visible.
5. If you see artificial colored lines on the screen, press the **s-key** to perform the calibration and save the values (_do not move the pattern for few seconds_).
6. To end the calibration proces and to stop the Python script, press the **q-key**.

> Each time you press the s-key, the calibration is carried out again and the values are overwritten in the file `src/camera_params.npz` (_if file exist_). 
> 
> To display the values at any time later, you can execute the Python script `dev/show_calibration.py`.

**Store camera params and show params:**

> **Important!** Before you execute the script `dev/get_calibration.py`, adapt the constants **WINDOW_WIDTH** and **WINDOW_HEIGHT** to your camera settings.

```shell
# run camera calibration
(.venv) $ python3 dev/get_calibration.py

# show camera values
(.venv) $ python3 dev/show_calibration.py
```

## Generate ArUco markers

> If you cannot print out the marker, use this [online marker generator](https://chev.me/arucogen/) on your mobile!

With the Python script `dev/generate_marker.py`, you can create your own ArUco markers. To follow the examples, you should print out markers with **ARUCO_MARKER_ID** `0`, `1` and `2`.

> Important are constants **ARUCO_DICT_ID** as well as **ARUCO_MARKER_ID** and **ARUCO_MARKER_SIZE**.
>
> **ARUCO_DICT_ID** select the ArUco Marker Set (_eq. DICT_4X4_100, DICT_6X6_50 or DICT_7X7_1000_).
>
> **ARUCO_MARKER_ID** select the ArUco marker id (_depends on ArUco Marker Set_).
>
> **ARUCO_MARKER_SIZE** set the size (_in pixels_) of ArUco markers.

The default of **ARUCO_DICT_ID** set is: `DICT_4X4_50`, which contains 50 predefined markers. The constant default value for **ARUCO_MARKER_ID** is `0`. For current **ARUCO_DICT_ID** the marker id's can be from `0` to `49`. The optimal value for **ARUCO_MARKER_SIZE** should be between `50` and `250`. Markers that are too small are harder to recognize.

**Generate markers:**

```shell
# run marker generation for id 0 / size 100 (default)
(.venv) $ python3 dev/generate_marker.py

# run marker generation for id 1 /size 100
(.venv) $ python3 dev/generate_marker.py --id 1 --size 100

# run marker generation for id 0 till 2 (example markers)
(.venv) $ for id in {0..2}; do python3 dev/generate_marker.py --id "$id" --size 100; done

# show created markers (optional)
(.venv) $ ls -la dev/markers/
```

Print out the marker(s) on paper, cut them and glue the printed paper onto cardboard (_for stabilization_).

## Run examples

> **Important!** Before you execute the example scripts, adapt the constants **WINDOW_WIDTH** and **WINDOW_HEIGHT** to your camera settings (_in each example file_).

| Example File    | Description                                                                                          |
|-----------------|------------------------------------------------------------------------------------------------------|
| `example_01.py` | Shows for each detected marker the respective ID.                                                    |
| `example_02.py` | Displays a letter from the alphabet for each detected marker (max. 26 letters).                      |
| `example_03.py` | Draws an arrow between each marker center and shows the distance in pixels.                          |
| `example_04.py` | Draws a line between marker centers and displays the distance in centimeters.                        |
| `example_05.py` | Projects a 3D cube on each detected marker.                                                          |
| `example_06.py` | Displays X, Y, Z axes on each detected marker.                                                       |
| `example_07.py` | Shows a scaled picture on each marker, based on the marker ID.                                       |
| `example_08.py` | Displays a scaled video loop on each marker, based on the marker ID.                                 |
| `example_09.py` | Shows a scaled picture between two specific markers, based on their marker ID combination.           |
| `example_10.py` | Displays a scaled video behind a circular mask created using all markers.                            |
| `example_11.py` | Projects filled cubes on markers, merging them if the distance between markers is below a set value. |
| `example_12.py` | Use marker id 0 as eraser and draw with all other markers lines on screen                            |
| `example_13.py` | For each marker the position can be saved (_m-key_) and difference is shown.                         |

> In few examples you still have to specify the length or height of the ArUco markers for constant: **MARKER_SIZE**. The measurement value must be provided in meters (_example: 3.5cm is 0.035_).
> 
> Measure one of the created ArUco markers and change the values for **MARKER_SIZE** in all required example files.
> 
> If you change the value for **ARUCO_DICT_ID**, you need to adapt the value in all example files too.

**Execute example:**

```shell
# execute example 01
(.venv) $ python3 example_01.py

# execute all examples
(.venv) $ for i in {01..13}; do python example_"$i".py; done
```

To close the window and to stop the Python script, press the **q-key** or **ESC-key**.

## Note

- Example images are generated with [perchance.org](https://perchance.org/ai-text-to-image-generator)
- Example videos downloaded from [pixabay.com](https://pixabay.com/)
- Example pattern downloaded from [github.com](https://github.com/opencv/opencv/blob/4.x/doc/pattern.png)
