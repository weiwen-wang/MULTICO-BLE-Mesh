# MULTICO-BLE-Mesh


**Mesh knowledge**

This file contains the overview of BLE Mesh, Mesh Profile and Mesh Model

**Weiwen's Thesis**

This file contains the theis and presentation about my work.



For the sensor part:

In sensor code, the paper introduce the mathematical process and the coding is based on it. The microphone should be connected to the raspberry pi with USB interfacce. And the get_device_index.py can be used to get the index of microphone. This index will be used in the testing_code.py. In testing_code.py, the voice is recorded and then processed to compute the feature vetor.

For the Mesh part:

Follow the instruction of Nodic document:

Nordic Semiconductor SDK for Mesh:
https://infocenter.nordicsemi.com/index.jsp?topic=%2Fcom.nordic.infocenter.meshsdk.v5.0.0%2Fmd_doc_user_guide_modules_provisioning_implementing.html

In the Getting started, it shows how to install the tool chain and building the stack.
https://infocenter.nordicsemi.com/index.jsp?topic=%2Fcom.nordic.infocenter.meshsdk.v5.0.0%2Fmd_doc_user_guide_modules_provisioning_implementing.html

I built the environment based in SEGGER Embedded Studio. You can also use CMake if you are familiar with the Linux.

I used the very simple example of generic_on_off to test the communication performance. But I tried to send signal between nRF52840 and Raspbery Pi with GPIO connection. If you want use this, don't forget connect the ground pin of the two board.

After flashing the project in nRF52840, you need a provisioner to build the mesh network. I use the nRF Mesh app on the phone, you can download it on either Apple store or Google shop. Raspberry pi can also be the provisioner but you need some operation to install bluez core in it. The details are in the **Raspberryprovision**

Some useful links:


https://docs.zephyrproject.org/latest/samples/classic.html
The Zephyr OS is based on a small-footprint kernel designed for use on resource-constrained and embedded systems, also suitable for nRF52840


https://devzone.nordicsemi.com/
The forum of Nordic Semiconductor, find the solutions of the problems with nRF52840.


