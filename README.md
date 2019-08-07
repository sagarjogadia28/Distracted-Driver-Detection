# Distracted-Driver-Detection

An android app captures drivers' images at predefined interval (e.g., 0.5s or 1s) and send API request. This API classify the image into following classes:
* texting
* talking on the phone
* drinking
* operating the radio
* talking to passenger
* hair and makeup
* reaching behind
* safe driving

If driver is caught distrated, the driver is alerted through cell's voice notification.

# Classification Model
Convolutional Neural Network VGG16 is used to train and classify images.
