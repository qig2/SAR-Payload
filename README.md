# SAR-Payload
# Instructions on the SAR Project (Software side)

## DJI Official Document
All prototypes are provided by DJI. Please firstly refer to [DJI Official Document](https://developer.dji.com/cn/payload-sdk/documentation/introduction/index.html) when there is a problem. 
Please also refer to the [DJI Developer Forum](https://bbs.dji.com/forum-79-1.html?from=developer).

## Required components
1. `CMake` `Python3` on Raspberry Pi
2. `Payload SDK` installation packet
3. `DJI Assistant 2` installation packet for Matrice 300 RTK
4. `Raspberry Pi 4B+` with `Raspberry Pi OS 32-bit`
5. `USB to tty converter`
6. `Skyport Development board`
7. `Skyport V2`
8. `DJI Matrice 300 RTK`

## Files
Files modified by us are listed below, please refer to the comments on individual files for more information. Files provided with comments are not in their real positions, please note that `Pay5/` is the root folder of the Payload SDK.

- Pay5/sample/api_sample/widget/test_widget.c
- Pay5/sample/api_sample/camera_media_emu/test_payload_cam_media.c
- Pay5/sample/platform/linux/manifold2/hal/hal_network.c
- Pay5/sample/platform/linux/manifold2/hal/hal_uart.c
- Pay5/sample/platform/linux/mainfold2/project/CMakeLists.txt
- Pay5/sample/platform/linux/mainfold2/project/build/SARimage_python.py
- Pay5/sample/platform/linux/mainfold2/project/build/wKA_fft_1d.py 
- Pay5/sample/platform/linux/mainfold2/project/application/app_info.h
- Pay5/sample/platform/linux/mainfold2/project/application/main.c
- Pay5/sample/platform/linux/mainfold2/project/application/psdk_config.h

## Connections between components
1. Raspberry Pi -> USB to `tty` converter -> `UART` -> `skyport` development board -> `skyport v2` -> drone
2. Raspberry Pi -> Ethernet Connection -> `skyport` development board -> `skyport v2` -> drone
![](DJI%20Skyport%20Development/IMG_3135.HEIC)

## First time setup
1. Apply for Business account.
2. Download `PSDK`, `DJI Assistant 2 `under developer account.
3. Copy the file to Raspberry Pi (Cautious when using a Mac, may encounter problem with `com.apple.quarantine` problem when `make`, recommend to download in Raspberry Pi).
4. Modify some files according to the website’s instructions [Tutorial on modifying the psdk_demo](https://www.freesion.com/article/2294962127/).
5. Follow the DJI Official document for PSDK [DJI PSDK Document](https://developer.dji.com/cn/payload-sdk/documentation/quickstart/run-the-sample.html).
6. Make sure the network port and serial port match with your own configuration. (`ls -l /dev/tty*` and `ls -l /dev/serial*` and `ifconfig`, example serial port `/dev/ttyUSB0` using tty converter, network port `eth0` for ethernet connection)
7. Install `DJI Assistant 2` on your PC, log in your DJI account and connect your drone with PC through the Type-C port on top of the drone.
8. Bind your device on `DJI Assistant 2`.
9. Run the following code on the Raspberry Pi.
```
> cd Pay5/sample/platform/linux/mainfold2/project/build
> cmake.. && make clean && make
> sudo ./SAR_module_adaption
```
10. The `Binding status` of the `DJI Skyport V2` should be `Bound`.

## Possible Errors
1. [PSDK init fail] May not register device on DJI Assistant 2.
2. [Network permission on `./psdk_demo`] Add `sudo` in front.

## Make Project
After modifying any file in the folder, please remake the whole project. 

Before Making the project, please make sure you have the following files in the `build` folder.

- data_subscription_data_in_main.txt
- SAR_start_timestamp.txt
- wKA_fft_1d.py 

Then run the following code in terminal.
```
> cd Pay5/sample/platform/linux/mainfold2/project/build
> cmake.. && make clean && make
> sudo ./SAR_module_adaption
```

## Rough estimation of the start/stop timestamp using python
Please run `find_5_m_s.py`.

# Contributions
- Qi GAO (qigao.17@intl.zju.edu.cn)
- Xiaoyu MA (xiaoyu.17@intl.zju.edu.cn)
- Kaiwei SUN (kaiwei.17@intl.zju.edu.cn)
- Zhizhan TANG (zhizhan.17@intl.zju.edu.cn)