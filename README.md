```bash
conda env create -f environment.yml -n SDP
conda activate SDP
cd server
python server.py

# copy the ip into main_v2.0.cpp
# upload the code to the sensor ESP32 using platformio

# upload the state machine code to the second ESP32 using platformio

# Unplug the computer from all ESP32
# Restart the server code
# Plug in the battery. 
# Everything should connect between the server and microcontroller, and the light should go solid on the microcontroller for when that happens.
```