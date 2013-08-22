LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)

#OPENCV_CAMERA_MODULES:=off
#OPENCV_INSTALL_MODULES:=off
#OPENCV_LIB_TYPE:=SHARED
include C:\adt-bundle-windows-x86-20130219\OpenCV-2.4.5-android-sdk\sdk\native\jni\OpenCV.mk

LOCAL_SRC_FILES  := DetectionBasedTracker.cpp
LOCAL_C_INCLUDES += $(LOCAL_PATH)
LOCAL_LDLIBS     += -llog -ldl
LOCAL_MODULE     := detection_based_tracker
include $(BUILD_SHARED_LIBRARY)