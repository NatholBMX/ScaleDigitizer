# ScaleDigitizer
The goal of this project is to extract digits from the seven digit display
of a bathroom scale to get the displayed weight. Having multiple videos available, one can average the weight
and save this to a database to track relative weight changes from week to week.

# Requirements
For running the project as is, following packages are needed:

* Python 3
* Numpy
* openCV
* skimage
* imutils

# What does it do?
We need videos of the bathroom scale, such as the following example:
![Unprocessed image](https://github.com/NatholBMX/ScaleDigitizer/blob/master/images/01.PNG)

We then process each image individually. First off, the region of interest(ROI) of the scale
will be calculated by applying a blur, edge detection and hough line transformation.
The result is a cropped image which then is taken under a 4-point transformation and a
threshold, resulting in following output:
![Thresholded image](https://github.com/NatholBMX/ScaleDigitizer/blob/master/images/02.PNG)

Next on, morphological dilation is applied to the threshold image to thicken the digits
and seperate them via openCV contour methods. The result can be seen here:
![Segmented digits](https://github.com/NatholBMX/ScaleDigitizer/blob/master/images/03.PNG)

By dividing the segments into sub-segments for a seven segment display of each digit,
one can simply count the number of pixel in relation the the total count of pixels of a sub-segment
to find out whether the seven-segment-part is activated or not. This way, a recognition of digits
is possible. Recognized digits can be displayed:
![Recognized digit](https://github.com/NatholBMX/ScaleDigitizer/blob/master/images/04.PNG)

Finally, a filtering of the recognized values is applied: the most occurrences of recognized digits
are taken as the recognized value for every video. These votings are converted to a single floating point
number and then all recognized numbers are averaged over the number of available videos inside the designated
folder.

Every average is saved to a database and compared to the previous entry to calculate a relative
difference in weight.

# Using the script
```python
python analyizeScale -video_path=PATH_TO_VIDEOS -database_name=NAME_OF_DATABASE -visualize=BOOL
```

Arguments are optionals, since we will look for videos under ````./videos/```` and the database
will be stored under ```./database/data``` by default. Results will not be visualized be default.

# References 
https://www.pyimagesearch.com/2017/02/13/recognizing-digits-with-opencv-and-python/

https://github.com/suyashkumar/seven-segment-ocr

https://www.kurokesu.com/main/2017/02/20/dumb-thermometer-gets-digital-output/

https://github.com/kazmiekr/GasPumpOCR.git

(Medium: https://hackernoon.com/building-a-gas-pump-scanner-with-opencv-python-ios-116fe6c9ae8b)

https://github.com/bikz05/digit-recognition

https://github.com/g-sari/pyautodigits

https://towardsdatascience.com/scanned-digits-recognition-using-k-nearest-neighbor-k-nn-d1a1528f0dea

# Special thanks
Thanks to [jwansek](https://github.com/jwansek) for contributing to the preprocessing.