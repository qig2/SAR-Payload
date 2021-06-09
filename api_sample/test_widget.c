/**
 ********************************************************************
 * @file    test_widget.c
 * @version V2.0.0
 * @date    2019/07/01
 * @brief
 *
 * @copyright (c) 2018-2019 DJI. All rights reserved.
 *
 * All information contained herein is, and remains, the property of DJI.
 * The intellectual and technical concepts contained herein are proprietary
 * to DJI and may be covered by U.S. and foreign patents, patents in process,
 * and protected by trade secret or copyright law.  Dissemination of this
 * information, including but not limited to data and other proprietary
 * material(s) incorporated within the information, in any form, is strictly
 * prohibited without the express written consent of DJI.
 *
 * If you receive this source code without DJI’s authorization, you may not
 * further disseminate the information, and you must immediately remove the
 * source code and notify DJI of its removal. DJI reserves the right to pursue
 * legal actions against you for any loss(es) or damage(s) caused by your
 * failure to do so.
 *
 *********************************************************************
 */

/* Includes ------------------------------------------------------------------*/
#include "test_widget.h"
#include <psdk_widget.h>
#include <psdk_logger.h>
#include <utils/util_misc.h>
#include <psdk_platform.h>
#include <stdio.h>
#include <wiringPi.h>
#include <Python.h>


#include "psdk_data_subscription.h"

#if !PSDK_ARCH_SYS_LINUX

#include "file_binary_array_list_en.h"

#endif

/* Private constants ---------------------------------------------------------*/
#define WIDGET_DIR_PATH_LEN_MAX         (256)
#define WIDGET_TASK_STACK_SIZE          (2048)
// #define RECORD_AUDIO
#define PYTHON_ON

/* Private types -------------------------------------------------------------*/

/* Private functions declaration ---------------------------------------------*/
static void *PsdkTest_WidgetTask(void *arg);

static T_PsdkReturnCode PsdkTestWidget_SetWidgetValue(E_PsdkWidgetType widgetType, uint32_t index, int32_t value,
                                                      void *userData);
static T_PsdkReturnCode PsdkTestWidget_GetWidgetValue(E_PsdkWidgetType widgetType, uint32_t index, int32_t *value,
                                                      void *userData);
// void widget_get_data_subscription(char* cur_time_stamp);
void widget_get_data_subscription(void);

void widget_get_data_subscription_end(void);




//void python_en(void);
static void *python_en(void *arg );
// static void * system_call(char* time_stamp);

static void * system_call(void *arg);

static void * widget_get_data_subscription_handler(void *arg);

static void * widget_get_data_subscription_end_handler(void *arg);


/* Private values ------------------------------------------------------------*/
static T_PsdkTaskHandle s_widgetTestThread;

static T_PsdkTaskHandle python_Thread;

static T_PsdkTaskHandle system_call_Thread;

static T_PsdkTaskHandle get_data_description_Thread = NULL;

static T_PsdkTaskHandle get_data_description_end_Thread = NULL;

static PyObject* pName = NULL;
static PyObject* pModule = NULL;
static PyObject* pDict = NULL;
static PyObject* pFunc = NULL;


static int counter = 0; // Used to generate new file names



#if PSDK_ARCH_SYS_LINUX
static bool s_isWidgetFileDirPathConfigured = false;
static char s_widgetFileDirPath[PSDK_FILE_PATH_SIZE_MAX] = {0};
#endif

static const T_PsdkWidgetHandlerListItem s_widgetHandlerList[] = {
    {0, PSDK_WIDGET_TYPE_BUTTON,        PsdkTestWidget_SetWidgetValue, PsdkTestWidget_GetWidgetValue, NULL},
    {1, PSDK_WIDGET_TYPE_BUTTON,        PsdkTestWidget_SetWidgetValue, PsdkTestWidget_GetWidgetValue, NULL},
    {2, PSDK_WIDGET_TYPE_LIST,          PsdkTestWidget_SetWidgetValue, PsdkTestWidget_GetWidgetValue, NULL},
    {3, PSDK_WIDGET_TYPE_SWITCH,        PsdkTestWidget_SetWidgetValue, PsdkTestWidget_GetWidgetValue, NULL},
    {4, PSDK_WIDGET_TYPE_SCALE,         PsdkTestWidget_SetWidgetValue, PsdkTestWidget_GetWidgetValue, NULL},
    {5, PSDK_WIDGET_TYPE_BUTTON,        PsdkTestWidget_SetWidgetValue, PsdkTestWidget_GetWidgetValue, NULL},
    {6, PSDK_WIDGET_TYPE_SCALE,         PsdkTestWidget_SetWidgetValue, PsdkTestWidget_GetWidgetValue, NULL},
    {7, PSDK_WIDGET_TYPE_INT_INPUT_BOX, PsdkTestWidget_SetWidgetValue, PsdkTestWidget_GetWidgetValue, NULL},
    {8, PSDK_WIDGET_TYPE_SWITCH,        PsdkTestWidget_SetWidgetValue, PsdkTestWidget_GetWidgetValue, NULL},
    {9, PSDK_WIDGET_TYPE_SWITCH,        PsdkTestWidget_SetWidgetValue, PsdkTestWidget_GetWidgetValue, NULL},
    {10, PSDK_WIDGET_TYPE_SWITCH,        PsdkTestWidget_SetWidgetValue, PsdkTestWidget_GetWidgetValue, NULL}
    //{9, PSDK_WIDGET_TYPE_LIST,          PsdkTestWidget_SetWidgetValue, PsdkTestWidget_GetWidgetValue, NULL},
};

static char *s_widgetTypeNameArray[] = {
    "Unknown",
    "Button",
    "Switch",
    "Scale",
    "List",
    "Int input box"
};

static const uint32_t s_widgetHandlerListCount = sizeof(s_widgetHandlerList) / sizeof(T_PsdkWidgetHandlerListItem);
static int32_t s_widgetValueList[sizeof(s_widgetHandlerList) / sizeof(T_PsdkWidgetHandlerListItem)] = {0};



/* Exported functions definition ---------------------------------------------*/
T_PsdkReturnCode PsdkTest_WidgetInit(void)
{
    T_PsdkReturnCode psdkStat;

    //Step 1 : Init PSDK Widget
    wiringPiSetup();
    psdkStat = PsdkWidget_Init();
    if (psdkStat != PSDK_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        PsdkLogger_UserLogError("Psdk test widget init error, stat = 0x%08llX", psdkStat);
        return psdkStat;
    }

#if PSDK_ARCH_SYS_LINUX
    //Step 2 : Set UI Config (Linux environment)

    char curFileDirPath[WIDGET_DIR_PATH_LEN_MAX];
    char tempPath[WIDGET_DIR_PATH_LEN_MAX];
    psdkStat = PsdkUserUtil_GetCurrentFileDirPath(__FILE__, WIDGET_DIR_PATH_LEN_MAX, curFileDirPath);
    if (psdkStat != PSDK_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        PsdkLogger_UserLogError("Get file current path error, stat = 0x%08llX", psdkStat);
        return psdkStat;
    }

    if (s_isWidgetFileDirPathConfigured == true) {
        snprintf(tempPath, WIDGET_DIR_PATH_LEN_MAX, "%swidget_file/en_big_screen", s_widgetFileDirPath);
    } else {
        snprintf(tempPath, WIDGET_DIR_PATH_LEN_MAX, "%swidget_file/en_big_screen", curFileDirPath);
    }

    //set default ui config path
    psdkStat = PsdkWidget_RegDefaultUiConfigByDirPath(tempPath);
    if (psdkStat != PSDK_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        PsdkLogger_UserLogError("Add default widget ui config error, stat = 0x%08llX", psdkStat);
        return psdkStat;
    }

    //set ui config for English language
    psdkStat = PsdkWidget_RegUiConfigByDirPath(PSDK_AIRCRAFT_INFO_MOBILE_APP_LANGUAGE_ENGLISH,
                                               PSDK_AIRCRAFT_INFO_MOBILE_APP_SCREEN_TYPE_BIG_SCREEN,
                                               tempPath);
    if (psdkStat != PSDK_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        PsdkLogger_UserLogError("Add widget ui config error, stat = 0x%08llX", psdkStat);
        return psdkStat;
    }

    //set ui config for Chinese language
    if (s_isWidgetFileDirPathConfigured == true) {
        snprintf(tempPath, WIDGET_DIR_PATH_LEN_MAX, "%swidget_file/cn_big_screen", s_widgetFileDirPath);
    } else {
        snprintf(tempPath, WIDGET_DIR_PATH_LEN_MAX, "%swidget_file/cn_big_screen", curFileDirPath);
    }

    psdkStat = PsdkWidget_RegUiConfigByDirPath(PSDK_AIRCRAFT_INFO_MOBILE_APP_LANGUAGE_CHINESE,
                                               PSDK_AIRCRAFT_INFO_MOBILE_APP_SCREEN_TYPE_BIG_SCREEN,
                                               tempPath);
    if (psdkStat != PSDK_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        PsdkLogger_UserLogError("Add widget ui config error, stat = 0x%08llX", psdkStat);
        return psdkStat;
    }
#else
    //Step 2 : Set UI Config (RTOS environment)

    T_PsdkWidgetBinaryArrayConfig enWidgetBinaryArrayConfig = {
        .binaryArrayCount = g_EnBinaryArrayCount,
        .fileBinaryArrayList = g_EnFileBinaryArrayList
    };

    //set default ui config
    psdkStat = PsdkWidget_RegDefaultUiConfigByBinaryArray(&enWidgetBinaryArrayConfig);
    if (psdkStat != PSDK_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        PsdkLogger_UserLogError("Add default widget ui config error, stat = 0x%08llX", psdkStat);
        return psdkStat;
    }

#endif

    //Step 3 : Set widget handler list
    psdkStat = PsdkWidget_RegHandlerList(s_widgetHandlerList, s_widgetHandlerListCount);
    if (psdkStat != PSDK_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        PsdkLogger_UserLogError("Set widget handler list error, stat = 0x%08llX", psdkStat);
        return psdkStat;
    }

    //Step 4 : Run widget api sample task
    if (PsdkOsal_TaskCreate(&s_widgetTestThread, PsdkTest_WidgetTask, "user_widget_task", WIDGET_TASK_STACK_SIZE,
                            NULL) != PSDK_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        PsdkLogger_UserLogError("Psdk widget test task create error.");
        return PSDK_ERROR_SYSTEM_MODULE_CODE_UNKNOWN;
    }
    
		

	#ifdef PYTHON_ON	

		Py_Initialize();
		if (!Py_IsInitialized()){
			printf("python init failed");
			Py_Finalize();
		}
		PyRun_SimpleString("import sys");
		//PyRun_SimpleString("import sdkaaa");
		PyRun_SimpleString("sys.argv = ['']");
		PyRun_SimpleString("sys.path.append('./')");
		
		// pName = PyUnicode_FromString("wKA_FMCW");
		//  pName = PyUnicode_FromString("sdkaaa");
		pName = PyUnicode_FromString("wKA_fft_1d");
		

		
		pModule = PyImport_Import(pName);
		//pModule = PyImport_Import("sdkaaa");
		
		PyRun_SimpleString("print(\"success\")");
		//Py_DECREF(pName);
		pDict = PyModule_GetDict(pModule);
		pFunc = PyDict_GetItemString(pDict,"testfunc");
		
	#endif	

	
	
    return PSDK_ERROR_SYSTEM_MODULE_CODE_SUCCESS;
}

#if PSDK_ARCH_SYS_LINUX
T_PsdkReturnCode PsdkTest_WidgetSetConfigFilePath(const char *path)
{
    memset(s_widgetFileDirPath, 0, sizeof(s_widgetFileDirPath));
    memcpy(s_widgetFileDirPath, path, USER_UTIL_MIN(strlen(path), sizeof(s_widgetFileDirPath) - 1));
    s_isWidgetFileDirPathConfigured = true;

    return PSDK_ERROR_SYSTEM_MODULE_CODE_SUCCESS;
}
#endif

#ifndef __CC_ARM
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-noreturn"
#pragma GCC diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wformat"
#endif

/* Private functions definition-----------------------------------------------*/

/*
void widget_get_data_subscription(char* cur_time_stamp) {
	T_PsdkReturnCode psdkStat;
    T_PsdkDataSubscriptionQuaternion quaternion = {0};
    T_PsdkDataSubscriptionVelocity velocity = {0};
    T_PsdkDataSubscriptiontTimestamp timestamp = {0};
    T_PsdkDataSubscriptionGpsPosition gpsPosition = {0};	
	PsdkOsal_TaskSleepMs(1000 / 1);
	psdkStat = PsdkDataSubscription_GetValueOfTopicWithTimestamp(PSDK_DATA_SUBSCRIPTION_TOPIC_QUATERNION,
																 (uint8_t *) &quaternion,
																 sizeof(T_PsdkDataSubscriptionQuaternion),
																 &timestamp);
	if (psdkStat != PSDK_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
		PsdkLogger_UserLogError("get value of topic quaternion error.");
	} else {
		PsdkLogger_UserLogDebug("timestamp: millisecond %u microsecond %u.", timestamp.millisecond,
								timestamp.microsecond);
		PsdkLogger_UserLogDebug("quaternion: %f %f %f %f.", quaternion.q0, quaternion.q1, quaternion.q2,
								quaternion.q3);
		printf("quaternion: %f %f %f %f.\n", quaternion.q0, quaternion.q1, quaternion.q2,
								quaternion.q3);
		FILE* fp = fopen("SAR_start_timestamp.txt", "a+");
		fprintf(fp, "timestamp: millisecond %u microsecond %u.\n", timestamp.millisecond,
								timestamp.microsecond);
		fclose(fp);
	}
	sprintf(cur_time_stamp, "arecord -D \"plughw:2,0\" -f S16_LE -r 48000 -c 2 -d 600 -t wav %u_%u.wav", timestamp.millisecond, timestamp.microsecond);
	 
}
*/

void widget_get_data_subscription(void) {
	T_PsdkReturnCode psdkStat;
    T_PsdkDataSubscriptionQuaternion quaternion = {0};
    T_PsdkDataSubscriptionVelocity velocity = {0};
    T_PsdkDataSubscriptiontTimestamp timestamp = {0};
    T_PsdkDataSubscriptionGpsPosition gpsPosition = {0};	
	PsdkOsal_TaskSleepMs(1000 / 1);
	psdkStat = PsdkDataSubscription_GetValueOfTopicWithTimestamp(PSDK_DATA_SUBSCRIPTION_TOPIC_QUATERNION,
																 (uint8_t *) &quaternion,
																 sizeof(T_PsdkDataSubscriptionQuaternion),
																 &timestamp);
	if (psdkStat != PSDK_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
		PsdkLogger_UserLogError("get value of topic quaternion error.");
	} else {
		PsdkLogger_UserLogDebug("timestamp: millisecond %u microsecond %u.", timestamp.millisecond,
								timestamp.microsecond);
		PsdkLogger_UserLogDebug("quaternion: %f %f %f %f.", quaternion.q0, quaternion.q1, quaternion.q2,
								quaternion.q3);
		printf("quaternion: %f %f %f %f.\n", quaternion.q0, quaternion.q1, quaternion.q2,
								quaternion.q3);
		FILE* fp = fopen("SAR_start_timestamp.txt", "a+");
		fprintf(fp, "record_%d\n", counter - 1);
		fprintf(fp, "timestamp: millisecond %u microsecond %u.\n", timestamp.millisecond,
								timestamp.microsecond);
		fclose(fp);
	}
	 
}

void widget_get_data_subscription_end(void) {
	T_PsdkReturnCode psdkStat;
    T_PsdkDataSubscriptionQuaternion quaternion = {0};
    T_PsdkDataSubscriptionVelocity velocity = {0};
    T_PsdkDataSubscriptiontTimestamp timestamp = {0};
    T_PsdkDataSubscriptionGpsPosition gpsPosition = {0};	
	PsdkOsal_TaskSleepMs(1000 / 1);
	psdkStat = PsdkDataSubscription_GetValueOfTopicWithTimestamp(PSDK_DATA_SUBSCRIPTION_TOPIC_QUATERNION,
																 (uint8_t *) &quaternion,
																 sizeof(T_PsdkDataSubscriptionQuaternion),
																 &timestamp);
	if (psdkStat != PSDK_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
		PsdkLogger_UserLogError("get value of topic quaternion error.");
	} else {
		PsdkLogger_UserLogDebug("timestamp: millisecond %u microsecond %u.", timestamp.millisecond,
								timestamp.microsecond);
		PsdkLogger_UserLogDebug("quaternion: %f %f %f %f.", quaternion.q0, quaternion.q1, quaternion.q2,
								quaternion.q3);
		printf("quaternion: %f %f %f %f.\n", quaternion.q0, quaternion.q1, quaternion.q2,
								quaternion.q3);
		FILE* fp = fopen("SAR_start_timestamp.txt", "a+");
		fprintf(fp, "record_%d\n", counter - 1);
		fprintf(fp, "timestamp: millisecond %u microsecond %u.\n", timestamp.millisecond,
								timestamp.microsecond);
		fclose(fp);
	}
	 
}


static void *python_en(void *arg){
	USER_UTIL_UNUSED(arg);
	

	PyEval_CallObject(pFunc,NULL);
	//PyObject_Call(pFunc,NULL,NULL);
	// PyRun_SimpleString("print(sys.version)");
	/*
	PyObject* numpy_module = PyImport_Import(PyUnicode_FromString("numpy"));
	if(!numpy_module){
		printf("no pmoudle\n");
		PyErr_Print();
	}
	
	PyObject* scipy_module = PyImport_Import(PyUnicode_FromString("scipy"));
	if(!scipy_module){
		printf("no pmoudle\n");
		PyErr_Print();
	}
	
	PyObject* math_module = PyImport_Import(PyUnicode_FromString("math"));
	if(!math_module){
		printf("no pmoudle\n");
		PyErr_Print();
	}
	*/
	//PyRun_SimpleString("import numpy as np");
	//PyRun_SimpleString("import scipy");
	//PyRun_SimpleString("from scipy.fftpack import fft, fftshift, ifft");
	//PyRun_SimpleString("import math");
	/*
	//PyObject* pName = PyUnicode_FromString("sdkaaa");
	PyObject* pName = PyUnicode_FromString("SARimage_python");
	
	printf("11111");
	PyObject* pModule = PyImport_Import(pName);
	if(! pModule){
		printf("no pmoudle\n");
		PyErr_Print();
	}
	*/
	/*
	printf("222222");
	PyObject* pDict = PyModule_GetDict(pModule);
	PyObject* pFunc = PyDict_GetItemString(pDict,"testfunc");
	printf("33333333");
	PyObject_CallObject(pFunc,NULL);
	*/
	

}

/*
static void * system_call(char* time_stamp) {
	// USER_UTIL_UNUSED(arg);
	// system("arecord -D \"plughw:2,0\" -f S16_LE -r 48000 -c 2 -d 600 -t wav interrupt.wav");
	printf("%s\n", time_stamp);
	time_stamp="arecord -D \"plughw:2,0\" -f S16_LE -r 48000 -c 2 -d 600 -t wav interrupt.wav";
	system(time_stamp);
}
*/


static void * system_call(void *arg) {
	USER_UTIL_UNUSED(arg);
	char cur_command[100];
	sprintf(cur_command, "arecord -D \"plughw:2,0\" -f S16_LE -r 48000 -c 2 -d 1200 -t wav ./record_%d.wav\n", counter);
	counter++;
	// system("arecord -D \"plughw:2,0\" -f S16_LE -r 48000 -c 2 -d 600 -t wav ./record.wav");
	system(cur_command);
}

static void * widget_get_data_subscription_handler(void *arg) {
	USER_UTIL_UNUSED(arg);
	widget_get_data_subscription();
}

static void * widget_get_data_subscription_end_handler(void *arg) {
	USER_UTIL_UNUSED(arg);
	widget_get_data_subscription_end();
}



static void *PsdkTest_WidgetTask(void *arg)
{
    char message[PSDK_WIDGET_FLOATING_WINDOW_MSG_MAX_LEN];
    uint32_t sysTimeMs = 0;
    T_PsdkReturnCode psdkStat;

    USER_UTIL_UNUSED(arg);

    while (1) {
        psdkStat = PsdkOsal_GetTimeMs(&sysTimeMs);
        if (psdkStat != PSDK_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
            PsdkLogger_UserLogError("Get system time ms error, stat = 0x%08llX", psdkStat);
        }
        snprintf(message, PSDK_WIDGET_FLOATING_WINDOW_MSG_MAX_LEN, "System time : %u ms", sysTimeMs);

        psdkStat = PsdkWidgetFloatingWindow_ShowMessage(message);
        if (psdkStat != PSDK_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
            PsdkLogger_UserLogError("Floating window show message error, stat = 0x%08llX", psdkStat);
        }

        PsdkOsal_TaskSleepMs(1000);
    }
}

#ifndef __CC_ARM
#pragma GCC diagnostic pop
#endif

static T_PsdkReturnCode PsdkTestWidget_SetWidgetValue(E_PsdkWidgetType widgetType, uint32_t index, int32_t value,
                                                      void *userData)
{
    USER_UTIL_UNUSED(userData);
	
    PsdkLogger_UserLogInfo("Set widget value, widgetType = %s, widgetIndex = %d ,widgetValue = %d",
                           s_widgetTypeNameArray[widgetType], index, value);
    s_widgetValueList[index] = value;
    pinMode (0, OUTPUT);
    pinMode (1, OUTPUT);
    char cur_time_stamp[100] = "";
    switch (index){
		case 8:
		if (value == 1) {
			printf("ADC Power On\n");
			digitalWrite(1, HIGH);
			digitalWrite(0, HIGH);
		} else {
			printf("ADC Power Off\n");
			digitalWrite(1, LOW);
			digitalWrite(0, LOW);
		}
		break;
		
		case 9:
		if (value == 1) {
			printf("Start Record\n");
			
			// widget_get_data_subscription(); // Does not work because it will generate two widget updates, should create a seperate thread.
			
			if (get_data_description_Thread == NULL) {
				PsdkOsal_TaskCreate(&get_data_description_Thread, widget_get_data_subscription_handler, "user_widget_get_data_subscription_call_task", 4096, NULL);
			}
			/*
			if (get_data_description_Thread != NULL) {
				PsdkOsal_TaskDestroy(get_data_description_Thread);
			}
			*/
			get_data_description_Thread = NULL;
			// system("arecord -D \"plughw:2,0\" -f S16_LE -r 48000 -c 2 -d 600 -t wav Real.wav"); // Does not work because it will block the PSDK main thread
			
			system("sudo killall -9 arecord");
			if (system_call_Thread == NULL) {
				PsdkOsal_TaskCreate(&system_call_Thread, system_call, "user_system_call_task", 4096, NULL);
			}
			
		} else {
			printf("Record Finished\n");
			// widget_get_data_subscription();
			get_data_description_end_Thread = NULL;
			if (get_data_description_end_Thread == NULL) {
				PsdkOsal_TaskCreate(&get_data_description_end_Thread, widget_get_data_subscription_end_handler, "user_widget_get_data_subscription_call_task", 4096, NULL);
			}
			/*
			if (get_data_description_end_Thread != NULL) {
				PsdkOsal_TaskDestroy(get_data_description_end_Thread);
			}
			*/
			get_data_description_end_Thread = NULL;
			
			
			if (system_call_Thread != NULL) {
				PsdkOsal_TaskDestroy(system_call_Thread);
			}
			system_call_Thread = NULL;
			system("sudo killall -9 arecord");	
		}
		break;
		
		case 10:
		if (value == 1) {
			printf("Start Synthesis\n");
			
			// Enable this when use the onboard synthesis algorithm
			#ifdef PYTHON_ON
				//system("python3 SARimage_python.py");
				//python_en();
				//if (PsdkPlatform_RegOsalHandler(&python_Thread) != PSDK_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
				//	printf("psdk register osal handler error");
				//}

	

				if (python_Thread==NULL){
					PsdkOsal_TaskCreate(&python_Thread, python_en, "user_python_task", 40960,NULL);
				}		
			#endif
			

		} else {
			printf("Synthesis Finished\n");	
			
			#ifdef PYTHON_ON	
				if (python_Thread!=NULL){
					PsdkOsal_TaskDestroy(python_Thread);
					python_Thread = NULL;
					
				}
				//Py_Finalize();
			#endif
			
		}
		break;
		default:
		break;
	}
    return PSDK_ERROR_SYSTEM_MODULE_CODE_SUCCESS;
}

static T_PsdkReturnCode PsdkTestWidget_GetWidgetValue(E_PsdkWidgetType widgetType, uint32_t index, int32_t *value,
                                                      void *userData)
{
    USER_UTIL_UNUSED(userData);
    USER_UTIL_UNUSED(widgetType);

    *value = s_widgetValueList[index];

    return PSDK_ERROR_SYSTEM_MODULE_CODE_SUCCESS;
}

/****************** (C) COPYRIGHT DJI Innovations *****END OF FILE****/