# face-off
Containerized mediapipe app for generation of face animations.

~~~
docker build -t face_off .
~~~

~~~
docker run -ti --rm -e DISPLAY=$DISPLAY --device=/dev/video0:/dev/video0 -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd)/mesh:/mesh --gpus all face_off
~~~