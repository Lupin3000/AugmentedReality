# Python: Augmented Reality

This repository is intended to introduce the topic of **Python: Augmented Reality** by ArUco marker detection.

Following details are explained and have examples:

- Camera calibration
- ArUco marker generation
- ArUco marker detection
- ArUco marker pose estimation

## Minimum requirements

- Python 3.7.x installed (_tested with Python 3.7.3 on UNIHIKER and 3.12 on macOS_)
- USB or Laptop camera

## Project setup

```shell
# clone the project to local
$ git clone https://github.com/Lupin3000/AugmentedReality.git

# change into cloned directory
$ cd AugmentedReality/
```

## Project structure

The final folders and file structure of the project (_if no calibration has yet been carried out and no markers have been generated_).

```shell
# list all files/folder (optional)
$ tree .
|____.gitignore
|____requirements.txt
|____example_01.py
|____example_02.py
|____example_03.py
|____example_05.py
|____example_06.py
|____example_07.py
|____dev
| |____img
| | |____pattern.png
| |____get_calibration.py
| |____show_calbraion.py
| |____generate_marker.py
|____src
| |____videos
| | |____video_0.mp4
| | |____video_1.mp4
| |____photos
| | |____monk_0.jpg
| | |____monk_1.jpg
| | |____treasure_1.jpg
| | |____treasure_2.jpg
```

**Project description**

- The `root` folder of the project contains the files: `.gitignore`, `requirements.txt` and `example_*.py`.
- Except for the `requirements.txt` file, which is used to install the [necessary Python modules/libraries](requirements.txt), all other Python files serve as examples for marker detection.
- The `dev/` folder contains Python scripts that support you, for example, with camera calibration and ArUco marker generation.
- In the `src/` folder you will find two images `src/photos/` and two videos `src/videos/` that are used for the AR examples.
- In the `dev/img/` subfolder you will find the file `pattern.png`. This pattern is needed to be printed out for camera calibration.
- When you create new markers using the Python script `dev/generate_marker.py`, the markers are saved as *.jpg into the new subfolder within the `dev/markers/`.
- If you have carried out a camera calibration, you will find the file `camera_params.npz` in the `src/` folder. This file will be loaded in the AR examples (_if available_).

## Prepare a local development environment

Various Python modules/libraries are used in this project. It is therefore recommended to use Python virtual environment. The necessary modules/libraries are listed in the `requirements.txt` file.

The next commands show how the Python virtual environment is created and the installation of required modules/libraries is carried out.

**Create virtualenv and install packages/modules (_commands_):**

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
2. After printing, measure the length or width of a single cube! Depending on the printer, this can vary slightly. Then convert the value into the unit of measurement meters (_for example: 2.4cm is 0.024m_). Enter the value for the constant **SQUARE_SIZE** in the Python script `dev/get_calibration.py`.
3. Provide good lighting for the area. Avoid strong shadows between the printed pattern and the camera. Also, avoid any light reflections on camera. 
4. Start the Python script `dev/get_calibration.py` and hold the pattern in front of the camera so that it is completely visible.
5. If you see artificial colored lines on the screen, press the **s-key** to perform the calibration and save the values (_do not move the pattern for few seconds_).
6. To end the calibration proces and to stop the Python script, press the **q-key**.

> Each time you press the s-key, the calibration is carried out again and the values are overwritten in the file `src/camera_params.npz` (_if file exist_). 
> 
> To display the values at any time later, you can execute the Python script `dev/show_calibration.py`.

**Store camera params and show params (_commands_):**

```shell
# run camera calibration
(.venv) $ python3 dev/get_calibration.py

# show camera values
(.venv) $ python3 dev/show_calibration.py
```

## Generate ArUco markers

With the Python script `dev/generate_marker.py`, you can create your own ArUco markers. To follow the examples, you should print out markers with **ARUCO_MARKER_ID** `0` and `1`.

> Important are constants **ARUCO_DICT_ID** as well as **ARUCO_MARKER_ID** and **ARUCO_MARKER_SIZE**.
>
> **ARUCO_DICT_ID** select the ArUco Marker Set (_eq. DICT_4X4_100, DICT_6X6_50 or DICT_7X7_1000_).
>
> **ARUCO_MARKER_ID** select the ArUco marker id (_depends on ArUco Marker Set_).
>
> **ARUCO_MARKER_SIZE** set the size (_in pixels_) of ArUco markers.

The default of **ARUCO_DICT_ID** set is: `DICT_4X4_50`, which contains 50 predefined markers. The constant default value for **ARUCO_MARKER_ID** is `0`. For current **ARUCO_DICT_ID** the marker id's can be from `0` to `49`. The optimal value for **ARUCO_MARKER_SIZE** should be between `50` and `250`. Markers that are too small are harder to recognize.

**Generate markers (_commands_):**

```shell
# run marker generation for id 0
(.venv) $ python3 dev/generate_marker.py --id 0 --size 100

# run marker generation for id 1
(.venv) $ python3 dev/generate_marker.py --id 1 --size 100

# show created markers (optional)
(.venv) $ ls -la dev/markers/
```

Print out the marker(s) on paper, cut them and glue the printed paper onto cardboard (_for stabilization_).

## Run examples

- `example_01.py` shows for each detected marker the respective ID.
- `example_02.py` shows for each detected marker a letter from alphabet.
- `example_03.py` shows the distance between two markers (_pixels, cm_).
- `example_05.py` shows a scaled picture on each marker.
- `example_06.py` shows a scaled video loop on each marker.
- `example_07.py` shows a scaled picture between two markers.

> In the examples you still have to specify the length or height of the ArUco markers in meters in the Python script constant: **MARKER_SIZE** (_example: 3.5cm is 0.035m_).
> 
> Measure one of the created ArUco markers and change the values for **MARKER_SIZE** in all example files.
> 
> If you change the value for **ARUCO_DICT_ID**, you need to adapt the value in all example files too.

**Execute examples (_commands_):**

```shell
# execute example 01
(.venv) $ python3 example_01.py

# execute example 02
(.venv) $ python3 example_02.py
```

To close the window and to stop the Python script, press the **q-key**.

## Note

- Example images are generated with [perchance.org](https://perchance.org/ai-text-to-image-generator)
- Example videos downloaded from [pixabay.com](https://pixabay.com/)
- Example pattern downloaded from [github.com](https://github.com/opencv/opencv/blob/4.x/doc/pattern.png)
