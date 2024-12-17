# Flame Classifier

Requirements:
* PyTorch installed with CUDA (learn more: https://pytorch.org/get-started/locally/)

Usage:
* Run live_predict.py and either type 'webcam' or path to video file.
* While the measurement is running, press 1, 2 or 3 to change the observed state (for validation purposes)
* Press 'q' to stop the measurement and 'e' to start the measurement. If the is no measurement running, press 'q' to close the programm

Improvements for 3D-models:
* Decrease tolerances for cam-mount and platform-mount for a tighter fit
* Increase length of the pipe im cam-rohr to accomodate for camera lens length when fully zoomed in
