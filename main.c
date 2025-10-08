/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include "stm32f4xx.h"
#include "lcd_stm32f4.h"
#include <stm32f4xx_hal_tim.h>
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
// TODO: Add values for below variables
#define NS 128     // Number of samples in LUT
#define TIM2CLK  16000000 // STM Clock frequency: Hint You might want to check the ioc file
#define F_SIGNAL 1	// Frequency of output analog signal
#define DEBOUNCE_DELAY 50  // milliseconds
#define NUM_WAVEFORMS 6

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
TIM_HandleTypeDef htim2;
TIM_HandleTypeDef htim3;
DMA_HandleTypeDef hdma_tim2_ch1;

/* USER CODE BEGIN PV */
// TODO: Add code for global variables, including LUTs
uint32_t Sin_LUT[NS] = {2047, 2148, 2249, 2350, 2450, 2548, 2646, 2742, 2836, 2929, 3019, 3107, 3192, 3275, 3354, 3431, 3504, 3573, 3639, 3700, 3758, 3812, 3861, 3906, 3946, 3981, 4012, 4038, 4059, 4076, 4087, 4093, 4094, 4091, 4082, 4068, 4049, 4026, 3997, 3964, 3926, 3884, 3837, 3785, 3730, 3670, 3606, 3539, 3468, 3393, 3315, 3234, 3150, 3063, 2974, 2883, 2789, 2694, 2597, 2499, 2400, 2300, 2199, 2098, 1996, 1895, 1794, 1694, 1595, 1497, 1400, 1305, 1211, 1120, 1031, 944, 860, 779, 701, 626, 555, 488, 424, 364, 309, 257, 210, 168, 130, 97, 68, 45, 26, 12, 3, 0, 1, 7, 18, 35, 56, 82, 113, 148, 188, 233, 282, 336, 394, 455, 521, 590, 663, 740, 819, 902, 987, 1075, 1165, 1258, 1352, 1448, 1546, 1644, 1744, 1845, 1946, 2047};
uint32_t Saw_LUT[NS] = {0, 32, 64, 96, 128, 161, 193, 225, 257, 290, 322, 354, 386, 419, 451, 483, 515, 548, 580, 612, 644, 677, 709, 741, 773, 806, 838, 870, 902, 935, 967, 999, 1031, 1064, 1096, 1128, 1160, 1193, 1225, 1257, 1289, 1322, 1354, 1386, 1418, 1450, 1483, 1515, 1547, 1579, 1612, 1644, 1676, 1708, 1741, 1773, 1805, 1837, 1870, 1902, 1934, 1966, 1999, 2031, 2063, 2095, 2128, 2160, 2192, 2224, 2257, 2289, 2321, 2353, 2386, 2418, 2450, 2482, 2515, 2547, 2579, 2611, 2644, 2676, 2708, 2740, 2772, 2805, 2837, 2869, 2901, 2934, 2966, 2998, 3030, 3063, 3095, 3127, 3159, 3192, 3224, 3256, 3288, 3321, 3353, 3385, 3417, 3450, 3482, 3514, 3546, 3579, 3611, 3643, 3675, 3708, 3740, 3772, 3804, 3837, 3869, 3901, 3933, 3966, 3998, 4030, 4062, 4095};
uint32_t Triangle_LUT[NS] = {0, 65, 130, 195, 260, 325, 390, 455, 520, 585, 650, 715, 780, 845, 910, 975, 1040, 1105, 1170, 1235, 1300, 1365, 1430, 1495, 1560, 1625, 1690, 1755, 1820, 1885, 1950, 2015, 2080, 2145, 2210, 2275, 2340, 2405, 2470, 2535, 2600, 2665, 2730, 2795, 2860, 2925, 2990, 3055, 3120, 3185, 3250, 3315, 3380, 3445, 3510, 3575, 3640, 3705, 3770, 3835, 3900, 3965, 4030, 4095, 4095, 4030, 3965, 3900, 3835, 3770, 3705, 3640, 3575, 3510, 3445, 3380, 3315, 3250, 3185, 3120, 3055, 2990, 2925, 2860, 2795, 2730, 2665, 2600, 2535, 2470, 2405, 2340, 2275, 2210, 2145, 2080, 2015, 1950, 1885, 1820, 1755, 1690, 1625, 1560, 1495, 1430, 1365, 1300, 1235, 1170, 1105, 1040, 975, 910, 845, 780, 715, 650, 585, 520, 455, 390, 325, 260, 195, 130, 65, 0};
uint32_t Piano_LUT[] = {1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1900 ,1901 ,1901 ,1901 ,1901 ,1901 ,1900 ,1900 ,1900 ,1901 ,1901 ,1900 ,1900 ,1901 ,1901 ,1901 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1901 ,1901 ,1901 ,1901 ,1900 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1900 ,1901 ,1901 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1902 ,1902 ,1902 ,1902 ,1902 ,1902 ,1901 ,1901 ,1901 ,1901 ,1901 ,1902 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1900 ,1900 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1902 ,1901 ,1901 ,1901 ,1901 ,1901 ,1900 ,1900 ,1899 ,1899 ,1900 ,1899 ,1899 ,1899 ,1899 ,1899 ,1900 ,1900 ,1900 ,1899 ,1899 ,1899 ,1900 ,1900 ,1900 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1900 ,1901 ,1901 ,1901 ,1901 ,1902 ,1901 ,1901 ,1901 ,1901 ,1902 ,1902 ,1901 ,1901 ,1901 ,1902 ,1902 ,1902 ,1902 ,1902 ,1902 ,1902 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1900 ,1900 ,1900 ,1900 ,1900 ,1899 ,1899 ,1899 ,1899 ,1899 ,1899 ,1899 ,1899 ,1900 ,1900 ,1900 ,1900 ,1901 ,1901 ,1901 ,1901 ,1900 ,1900 ,1901 ,1902 ,1902 ,1901 ,1900 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1903 ,1903 ,1902 ,1901 ,1901 ,1901 ,1900 ,1901 ,1901 ,1901 ,1901 ,1900 ,1900 ,1901 ,1900 ,1900 ,1900 ,1900 ,1899 ,1899 ,1899 ,1899 ,1899 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1899 ,1899 ,1899 ,1899 ,1899 ,1900 ,1900 ,1900 ,1899 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1899 ,1899 ,1899 ,1900 ,1900 ,1900 ,1899 ,1900 ,1899 ,1899 ,1900 ,1901 ,1901 ,1902 ,1903 ,1903 ,1903 ,1903 ,1902 ,1903 ,1903 ,1903 ,1902 ,1902 ,1902 ,1902 ,1902 ,1902 ,1902 ,1902 ,1902 ,1902 ,1902 ,1902 ,1902 ,1903 ,1903 ,1902 ,1902 ,1902 ,1902 ,1902 ,1901 ,1901 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1901 ,1901 ,1900 ,1900 ,1901 ,1902 ,1903 ,1903 ,1903 ,1903 ,1904 ,1904 ,1904 ,1903 ,1903 ,1902 ,1902 ,1902 ,1901 ,1901 ,1901 ,1902 ,1901 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1901 ,1901 ,1901 ,1901 ,1902 ,1902 ,1901 ,1901 ,1902 ,1901 ,1901 ,1901 ,1900 ,1900 ,1900 ,1901 ,1900 ,1900 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1902 ,1901 ,1901 ,1901 ,1901 ,1902 ,1903 ,1902 ,1902 ,1902 ,1902 ,1902 ,1901 ,1901 ,1902 ,1902 ,1902 ,1901 ,1901 ,1900 ,1901 ,1901 ,1901 ,1901 ,1901 ,1900 ,1900 ,1900 ,1900 ,1899 ,1899 ,1899 ,1900 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1902 ,1902 ,1902 ,1901 ,1901 ,1901 ,1902 ,1902 ,1901 ,1901 ,1901 ,1902 ,1902 ,1902 ,1902 ,1901 ,1901 ,1901 ,1900 ,1900 ,1899 ,1899 ,1899 ,1900 ,1900 ,1900 ,1899 ,1900 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1902 ,1901 ,1901 ,1901 ,1902 ,1903 ,1903 ,1902 ,1902 ,1902 ,1903 ,1903 ,1902 ,1901 ,1901 ,1901 ,1901 ,1901 ,1902 ,1901 ,1902 ,1902 ,1901 ,1900 ,1900 ,1900 ,1901 ,1901 ,1901 ,1901 ,1901 ,1902 ,1902 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1900 ,1899 ,1899 ,1900 ,1899 ,1899 ,1898 ,1898 ,1898 ,1898 ,1897 ,1898 ,1899 ,1899 ,1899 ,1898 ,1899 ,1901 ,1901 ,1901 ,1900 ,1900 ,1901 ,1902 ,1902 ,1901 ,1901 ,1903 ,1903 ,1903 ,1903 ,1902 ,1902 ,1901 ,1900 ,1900 ,1900 ,1901 ,1902 ,1902 ,1902 ,1902 ,1902 ,1902 ,1901 ,1899 ,1898 ,1899 ,1899 ,1899 ,1899 ,1899 ,1899 ,1899 ,1900 ,1901 ,1902 ,1902 ,1902 ,1901 ,1900 ,1899 ,1901 ,1901 ,1900 ,1899 ,1900 ,1900 ,1901 ,1900 ,1901 ,1901 ,1901 ,1901 ,1901 ,1901 ,1902 ,1903 ,1902 ,1902 ,1901 ,1901 ,1901 ,1901 ,1900 ,1900 ,1900 ,1900 ,1900 ,1901 ,1902 ,1903 ,1904 ,1905 ,1904 ,1902 ,1902 ,1902 ,1902 ,1903 ,1903 ,1903 ,1904 ,1903 ,1903 ,1902 ,1901 ,1901 ,1902 ,1902 ,1902 ,1903 ,1902 ,1903 ,1902 ,1902 ,1901 ,1900 ,1902 ,1900 ,1900 ,1901 ,1899 ,1899 ,1899 ,1900 ,1901 ,1899 ,1902 ,1902 ,1902 ,1901 ,1901 ,1900 ,1900 ,1901 ,1899 ,1901 ,1902 ,1902 ,1905 ,1903 ,1902 ,1903 ,1901 ,1904 ,1902 ,1900 ,1901 ,1898 ,1900 ,1899 ,1898 ,1899 ,1897 ,1900 ,1901 ,1901 ,1903 ,1902 ,1902 ,1901 ,1900 ,1900 ,1900 ,1899 ,1900 ,1900 ,1900 ,1902 ,1902 ,1899 ,1899 ,1899 ,1899 ,1899 ,1899 ,1899 ,1900 ,1900 ,1902 ,1902 ,1899 ,1900 ,1902 ,1903 ,1902 ,1900 ,1901 ,1902 ,1901 ,1901 ,1901 ,1903 ,1902 ,1899 ,1899 ,1900 ,1901 ,1901 ,1901 ,1903 ,1903 ,1902 ,1903 ,1902 ,1899 ,1899 ,1899 ,1900 ,1899 ,1899 ,1900 ,1900 ,1902 ,1903 ,1902 ,1901 ,1902 ,1903 ,1899 ,1890 ,1877 ,1855 ,1825 ,1787 ,1738 ,1681 ,1615 ,1540 ,1456 ,1362 ,1260 ,1153 ,1038 ,918 ,795 ,669 ,540 ,411 ,282 ,167 ,105 ,81 ,46 ,20 ,10 ,0 ,0 ,7 ,25 ,47 ,74 ,112 ,154 ,201 ,252 ,314 ,378 ,443 ,518 ,592 ,670 ,752 ,836 ,920 ,996 ,1083 ,1181 ,1272 ,1351 ,1444 ,1575 ,1692 ,1726 ,1733 ,1784 ,1828 ,1892 ,1965 ,1987 ,2011 ,2051 ,2090 ,2119 ,2145 ,2235 ,2333 ,2379 ,2421 ,2437 ,2396 ,2369 ,2398 ,2391 ,2287 ,2183 ,2151 ,2123 ,1997 ,1841 ,1729 ,1625 ,1623 ,1668 ,1626 ,1610 ,1666 ,1752 ,1770 ,1724 ,1718 ,1709 ,1715 ,1713 ,1636 ,1510 ,1416 ,1393 ,1315 ,1153 ,1015 ,979 ,983 ,952 ,969 ,1002 ,1017 ,1051 ,1090 ,1190 ,1322 ,1413};
uint32_t Guitar_LUT[] = {2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2187 ,2187 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2187 ,2187 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2187 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2187 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2187 ,2187 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2187 ,2187 ,2187 ,2187 ,2187 ,2186 ,2186 ,2186 ,2187 ,2187 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2187 ,2187 ,2186 ,2186 ,2187 ,2188 ,2188 ,2187 ,2187 ,2187 ,2187 ,2187 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2185 ,2186 ,2186 ,2186 ,2186 ,2186 ,2186 ,2187 ,2187 ,2187 ,2186 ,2186 ,2186 ,2188 ,2188 ,2187 ,2186 ,2186 ,2187 ,2186 ,2185 ,2185 ,2186 ,2188 ,2188 ,2187 ,2186 ,2186 ,2187 ,2188 ,2187 ,2187 ,2187 ,2187 ,2186 ,2185 ,2185 ,2186 ,2186 ,2186 ,2188 ,2188 ,2188 ,2186 ,2185 ,2185 ,2186 ,2187 ,2186 ,2184 ,2184 ,2186 ,2187 ,2186 ,2185 ,2185 ,2187 ,2189 ,2189 ,2187 ,2185 ,2185 ,2187 ,2187 ,2186 ,2185 ,2185 ,2186 ,2186 ,2186 ,2186 ,2188 ,2190 ,2190 ,2189 ,2188 ,2188 ,2188 ,2187 ,2185 ,2184 ,2186 ,2188 ,2187 ,2185 ,2185 ,2187 ,2190 ,2190 ,2188 ,2186 ,2187 ,2188 ,2190 ,2188 ,2184 ,2183 ,2186 ,2187 ,2186 ,2184 ,2183 ,2184 ,2187 ,2190 ,2189 ,2187 ,2187 ,2187 ,2186 ,2183 ,2183 ,2185 ,2186 ,2186 ,2187 ,2189 ,2188 ,2185 ,2184 ,2186 ,2191 ,2191 ,2188 ,2183 ,2183 ,2185 ,2185 ,2180 ,2175 ,2178 ,2182 ,2181 ,2176 ,2172 ,2174 ,2179 ,2185 ,2189 ,2191 ,2192 ,2195 ,2198 ,2202 ,2206 ,2208 ,2207 ,2208 ,2213 ,2219 ,2219 ,2214 ,2209 ,2206 ,2200 ,2187 ,2168 ,2155 ,2148 ,2131 ,2103 ,2082 ,2079 ,2090 ,2105 ,2107 ,2103 ,2107 ,2129 ,2160 ,2182 ,2188 ,2187 ,2186 ,2183 ,2178 ,2178 ,2193 ,2217 ,2226 ,2214 ,2215 ,2253 ,2299 ,2312 ,2293 ,2278 ,2283 ,2269 ,2206 ,2136 ,2101};
uint32_t Drum_LUT[] = {2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2046 ,2046 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2046 ,2046 ,2046 ,2046 ,2046 ,2047 ,2047 ,2047 ,2046 ,2047 ,2048 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2046 ,2046 ,2046 ,2046 ,2047 ,2048 ,2048 ,2048 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2048 ,2047 ,2047 ,2048 ,2048 ,2047 ,2046 ,2047 ,2048 ,2047 ,2046 ,2046 ,2047 ,2047 ,2048 ,2048 ,2048 ,2048 ,2047 ,2046 ,2047 ,2047 ,2047 ,2047 ,2048 ,2049 ,2050 ,2049 ,2047 ,2046 ,2045 ,2045 ,2046 ,2048 ,2048 ,2047 ,2047 ,2047 ,2046 ,2046 ,2046 ,2046 ,2048 ,2049 ,2049 ,2048 ,2048 ,2048 ,2048 ,2047 ,2046 ,2046 ,2047 ,2049 ,2050 ,2048 ,2046 ,2046 ,2046 ,2045 ,2044 ,2045 ,2047 ,2049 ,2048 ,2046 ,2047 ,2046 ,2046 ,2048 ,2048 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2047 ,2046 ,2047 ,2049 ,2048 ,2046 ,2044 ,2044 ,2048 ,2050 ,2048 ,2047 ,2048 ,2048 ,2047 ,2045 ,2044 ,2046 ,2047 ,2048 ,2049 ,2048 ,2048 ,2048 ,2047 ,2046 ,2046 ,2048 ,2050 ,2050 ,2047 ,2046 ,2049 ,2049 ,2047 ,2046 ,2046 ,2048 ,2050 ,2048 ,2046 ,2046 ,2049 ,2049 ,2048 ,2048 ,2050 ,2050 ,2048 ,2045 ,2045 ,2046 ,2047 ,2046 ,2045 ,2044 ,2044 ,2047 ,2050 ,2050 ,2048 ,2047 ,2046 ,2049 ,2051 ,2049 ,2046 ,2046 ,2047 ,2047 ,2046 ,2044 ,2043 ,2046 ,2049 ,2047 ,2047 ,2049 ,2051 ,2051 ,2049 ,2048 ,2048 ,2048 ,2046 ,2046 ,2047 ,2046 ,2046 ,2047 ,2049 ,2049 ,2048 ,2048 ,2048 ,2048 ,2046 ,2044 ,2044 ,2046 ,2049 ,2050 ,2049 ,2048 ,2047 ,2047 ,2048 ,2049 ,2047 ,2044 ,2044 ,2045 ,2046 ,2046 ,2044 ,2044 ,2045 ,2046 ,2048 ,2049 ,2048 ,2047 ,2047 ,2047 ,2046 ,2048 ,2050 ,2050 ,2049 ,2048 ,2046 ,2046 ,2047 ,2047 ,2046 ,2043 ,2042 ,2045 ,2047 ,2046 ,2045 ,2045 ,2046 ,2048 ,2050 ,2050 ,2048 ,2045 ,2043 ,2044 ,2045 ,2045 ,2046 ,2047 ,2048 ,2049 ,2048 ,2045 ,2046 ,2047 ,2044 ,2043 ,2045 ,2045 ,2045 ,2046 ,2048 ,2051 ,2050 ,2047 ,2050 ,2052 ,2048 ,2044 ,2044 ,2045 ,2046 ,2045 ,2043 ,2043 ,2046 ,2050 ,2049 ,2046 ,2046 ,2047 ,2051 ,2054 ,2051 ,2047 ,2046 ,2045 ,2045 ,2046 ,2045 ,2042 ,2040 ,2042 ,2046 ,2047 ,2047 ,2048 ,2050 ,2049 ,2046 ,2043 ,2047 ,2050 ,2048 ,2047 ,2044 ,2044 ,2049 ,2048 ,2043 ,2044 ,2050 ,2052 ,2051 ,2048 ,2046 ,2046 ,2047 ,2047 ,2049 ,2050 ,2051 ,2052 ,2048 ,2045 ,2044 ,2046 ,2048 ,2047 ,2045 ,2049 ,2054 ,2052 ,2046 ,2043 ,2043 ,2047 ,2050 ,2047 ,2045 ,2046 ,2047 ,2048 ,2048 ,2047 ,2047 ,2045 ,2046 ,2050 ,2050 ,2047 ,2047 ,2046 ,2044 ,2048 ,2051 ,2051 ,2048 ,2045 ,2044 ,2045 ,2042 ,2040 ,2045 ,2050 ,2051 ,2047 ,2044 ,2043 ,2045 ,2046 ,2047 ,2047 ,2048 ,2051 ,2054 ,2054 ,2051 ,2050 ,2047 ,2046 ,2047 ,2047 ,2047 ,2050 ,2051 ,2048 ,2044 ,2041 ,2040 ,2042 ,2047 ,2050 ,2050 ,2049 ,2049 ,2050 ,2050 ,2048 ,2044 ,2045 ,2048 ,2048 ,2049 ,2051 ,2051 ,2049 ,2047 ,2044 ,2046 ,2050 ,2047 ,2042 ,2044 ,2050 ,2050 ,2048 ,2049 ,2050 ,2048 ,2048 ,2049 ,2048 ,2046 ,2045 ,2046 ,2049 ,2053 ,2054 ,2051 ,2047 ,2047 ,2047 ,2047 ,2047 ,2046 ,2047 ,2050 ,2050 ,2049 ,2049 ,2049 ,2050 ,2049 ,2048 ,2051 ,2053 ,2050 ,2048 ,2052 ,2052 ,2050 ,2048 ,2048 ,2049 ,2047 ,2041 ,2042 ,2047 ,2048 ,2048 ,2049 ,2047 ,2047 ,2050 ,2051 ,2049 ,2049 ,2050 ,2052 ,2051 ,2049 ,2047 ,2047 ,2047 ,2049 ,2048 ,2047 ,2046 ,2046 ,2045 ,2044 ,2042 ,2043 ,2046 ,2047 ,2047 ,2050 ,2049 ,2048 ,2047 ,2045 ,2043 ,2044 ,2046 ,2049 ,2053 ,2054 ,2049 ,2047 ,2046 ,2044 ,2044 ,2046 ,2047 ,2050 ,2051 ,2051 ,2051 ,2052 ,2051 ,2047 ,2042 ,2044 ,2050 ,2052 ,2049 ,2047 ,2048 ,2051 ,2052 ,2050 ,2048 ,2044 ,2044 ,2049 ,2049 ,2045 ,2042 ,2043 ,2049 ,2052 ,2046 ,2042 ,2045 ,2048 ,2047 ,2045 ,2045 ,2046 ,2047 ,2048 ,2044 ,2039 ,2043 ,2048 ,2048 ,2048 ,2051 ,2052 ,2050 ,2048 ,2049 ,2050 ,2048 ,2046 ,2045 ,2046 ,2047 ,2047 ,2047 ,2049 ,2049 ,2050 ,2048 ,2043 ,2041 ,2046 ,2050 ,2049 ,2046 ,2046 ,2045 ,2044 ,2044 ,2044 ,2045 ,2047 ,2047 ,2046 ,2044 ,2046 ,2053 ,2056 ,2049 ,2042 ,2045 ,2051 ,2052 ,2047 ,2041 ,2042 ,2046 ,2047 ,2045 ,2046 ,2049 ,2050 ,2045 ,2042 ,2047 ,2053 ,2051 ,2043 ,2040 ,2043 ,2046 ,2048 ,2049 ,2046 ,2044 ,2047 ,2048 ,2046 ,2042 ,2040 ,2043 ,2049 ,2052 ,2051 ,2049 ,2050 ,2050 ,2050 ,2052 ,2049 ,2045 ,2046 ,2047 ,2044 ,2046 ,2052 ,2052 ,2047 ,2046 ,2050 ,2051 ,2049 ,2049 ,2053 ,2054 ,2051 ,2049 ,2050 ,2050 ,2046 ,2045 ,2048 ,2051 ,2050 ,2048 ,2048 ,2048 ,2043 ,2044 ,2048 ,2049 ,2050 ,2052 ,2052 ,2052 ,2052 ,2051 ,2049 ,2043 ,2042 ,2048 ,2052 ,2054 ,2058 ,2053 ,2046 ,2049 ,2052 ,2048 ,2048 ,2048 ,2046 ,2046 ,2050 ,2052 ,2051 ,2051 ,2048 ,2044 ,2042 ,2046 ,2052 ,2052 ,2049 ,2050 ,2048 ,2042 ,2042 ,2051 ,2052 ,2048 ,2046 ,2045 ,2044 ,2046 ,2047 ,2045 ,2043 ,2041 ,2042 ,2047 ,2047 ,2043 ,2045 ,2049 ,2051 ,2053 ,2049 ,2047 ,2051 ,2055 ,2054 ,2053 ,2049 ,2045 ,2043 ,2039 ,2042 ,2052 ,2050 ,2043 ,2046 ,2053 ,2051 ,2047 ,2046 ,2047 ,2050 ,2050 ,2041 ,2040 ,2049 ,2049 ,2042 ,2042 ,2047 ,2052 ,2054 ,2051 ,2052 ,2056 ,2050 ,2044 ,2045 ,2044 ,2043 ,2046 ,2046 ,2047 ,2053 ,2055 ,2054 ,2048 ,2037 ,2036 ,2044 ,2045 ,2042 ,2046 ,2050 ,2049 ,2044 ,2034 ,2035 ,2045 ,2053 ,2056 ,2043 ,2031 ,2043 ,2054 ,2048 ,2050 ,2044 ,2031 ,2039 ,2044 ,2040 ,2048 ,2047 ,2043 ,2049 ,2048 ,2060 ,2058 ,2025 ,2052 ,2039 ,1844 ,1744 ,1930 ,2077 ,1951 ,1694 ,1546 ,1677 ,1909 ,1918 ,1825 ,1881 ,1939 ,1840 ,1751 ,1855 ,2049 ,2076 ,1911 ,1777 ,1770 ,1875 ,2054 ,2082 ,1851 ,1635 ,1671 ,1982 ,2420 ,2537 ,2196 ,1967 ,2085 ,2133 ,2066 ,2118 ,2168 ,2189 ,2355 ,2519 ,2459 ,2219 ,1913 ,1694 ,1677 ,1763 ,1845 ,1903 ,1890 ,1873 ,1911 ,1800 ,1484 ,1287 ,1345 ,1496 ,1677 ,1865 ,1931 ,1826 ,1638 ,1486 ,1399 ,1383 ,1598 ,1983 ,2110 ,1977 ,2003 ,2156 ,2178 ,2136 ,2096 ,2020 ,1957 ,1997 ,2177 ,2291 ,2121 ,1944 ,2026 ,2168 ,2309 ,2542 ,2659 ,2532 ,2383 ,2381 ,2383 ,2218 ,2114 ,2329 ,2549 ,2491 ,2367 ,2339 ,2367 ,2310 ,1980 ,1574 ,1528 ,1866 ,2239 ,2257 ,2003 ,1975 ,2024 ,1702 ,1461 ,1715 ,2034 ,2159 ,2149 ,1977 ,1822 ,1789 ,1778 ,1889 ,2072};

uint32_t* waveform_LUTs[NUM_WAVEFORMS] = {Sin_LUT, Saw_LUT, Triangle_LUT,Piano_LUT,Drum_LUT, Guitar_LUT };

char* waveform_names[NUM_WAVEFORMS] = {"SINE WAVE", "SAWTOOTH WAVE", "TRIANGLE WAVE","PIANO WAVE","DRUM WAVE","GUITAR WAVE"};

uint8_t current_waveform = 0;
uint32_t last_button_press = 0;


// TODO: Equation to calculate TIM2_Ticks
uint32_t TIM2_Ticks = TIM2CLK / (F_SIGNAL *NS);             // ticks between samples (will be computed at runtime)
uint32_t DestAddress = (uint32_t) & (TIM3->CCR3); // DMA destination (CCR3)


/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_DMA_Init(void);
static void MX_TIM2_Init(void);
static void MX_TIM3_Init(void);
/* USER CODE BEGIN PFP */
void EXTI0_IRQHandler(void);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_TIM2_Init();
  MX_TIM3_Init();
  /* USER CODE BEGIN 2 */
  // TODO: Start TIM3 in PWM mode on channel 3
  if (HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_3) != HAL_OK)
  {
      Error_Handler();   // If PWM start fails, handle error
  }
  // TODO: Start TIM2 in Output Compare (OC) mode on channel 1
  if (HAL_TIM_OC_Start(&htim2, TIM_CHANNEL_1) != HAL_OK)
  {
      Error_Handler();   // If OC start fails, stop and debug
  }

  // TODO: Start DMA in IT mode on TIM2->CH1. Source is LUT and Dest is TIM3->CCR3; start with Sine LUT
  HAL_DMA_Start_IT(&hdma_tim2_ch1,
    						 (uint32_t)Sin_LUT,
    						 (uint32_t)&(TIM3->CCR3),
    						 NS);
  // TODO: Write current waveform to LCD(Sine is the first waveform)
  init_LCD();
  lcd_putstring("SINE WAVE");
  // TODO: Enable DMA (start transfer from LUT to CCR)
  __HAL_TIM_ENABLE_DMA(&htim2, TIM_DMA_CC1);
  
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE3);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_NONE;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_HSI;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_0) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief TIM2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM2_Init(void)
{

  /* USER CODE BEGIN TIM2_Init 0 */

  /* USER CODE END TIM2_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};

  /* USER CODE BEGIN TIM2_Init 1 */

  /* USER CODE END TIM2_Init 1 */
  htim2.Instance = TIM2;
  htim2.Init.Prescaler = 0;
  htim2.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim2.Init.Period = TIM2_Ticks - 1;
  htim2.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim2.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim2) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim2, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_OC_Init(&htim2) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim2, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_TIMING;
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_OC_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM2_Init 2 */
  /* TIM2_CH1 DMA Init */
  __HAL_RCC_DMA1_CLK_ENABLE();

  hdma_tim2_ch1.Instance = DMA1_Stream5;
  hdma_tim2_ch1.Init.Channel = DMA_CHANNEL_3;         // TIM2_CH1 is on channel 3
  hdma_tim2_ch1.Init.Direction = DMA_MEMORY_TO_PERIPH; // Memory -> TIM3->CCR3
  hdma_tim2_ch1.Init.PeriphInc = DMA_PINC_DISABLE;    // Peripheral address fixed
  hdma_tim2_ch1.Init.MemInc = DMA_MINC_ENABLE;        // Memory address increments
  hdma_tim2_ch1.Init.PeriphDataAlignment = DMA_PDATAALIGN_WORD;
  hdma_tim2_ch1.Init.MemDataAlignment = DMA_MDATAALIGN_WORD;
  hdma_tim2_ch1.Init.Mode = DMA_CIRCULAR;            // Repeat LUT automatically
  hdma_tim2_ch1.Init.Priority = DMA_PRIORITY_HIGH;
  hdma_tim2_ch1.Init.FIFOMode = DMA_FIFOMODE_DISABLE;

  if (HAL_DMA_Init(&hdma_tim2_ch1) != HAL_OK)
  {
      Error_Handler();
  }

  /* Link DMA handle to TIM2 handle */
  __HAL_LINKDMA(&htim2, hdma[TIM_DMA_ID_CC1], hdma_tim2_ch1);
  /* USER CODE END TIM2_Init 2 */

}

/**
  * @brief TIM3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM3_Init(void)
{

  /* USER CODE BEGIN TIM3_Init 0 */

  /* USER CODE END TIM3_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};

  /* USER CODE BEGIN TIM3_Init 1 */

  /* USER CODE END TIM3_Init 1 */
  htim3.Instance = TIM3;
  htim3.Init.Prescaler = 0;
  htim3.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim3.Init.Period = 4095; // 12-bit resolution
  htim3.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim3.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim3) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim3, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_Init(&htim3) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim3, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_3) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM3_Init 2 */

  /* USER CODE END TIM3_Init 2 */
  HAL_TIM_MspPostInit(&htim3);

}

/**
  * Enable DMA controller clock
  */
static void MX_DMA_Init(void)
{

  /* DMA controller clock enable */
  __HAL_RCC_DMA1_CLK_ENABLE();

  /* DMA interrupt init */
  /* DMA1_Stream5_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA1_Stream5_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA1_Stream5_IRQn);

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  // -------------------------------
  // LCD pins configuration
  // -------------------------------
  // Configure PC14 (RS) and PC15 (E) as output push-pull
  GPIO_InitStruct.Pin = GPIO_PIN_14 | GPIO_PIN_15;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

  // Configure PB8 (D4) and PB9 (D5) as output push-pull
  GPIO_InitStruct.Pin = GPIO_PIN_8 | GPIO_PIN_9;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  // Configure PA12 (D6) and PA15 (D7) as output push-pull
  GPIO_InitStruct.Pin = GPIO_PIN_12 | GPIO_PIN_15;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  // Set all LCD pins LOW initially
  HAL_GPIO_WritePin(GPIOC, GPIO_PIN_14 | GPIO_PIN_15, GPIO_PIN_RESET);
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_8 | GPIO_PIN_9, GPIO_PIN_RESET);
  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_12 | GPIO_PIN_15, GPIO_PIN_RESET);


  // -------------------------------
  // Button0 configuration (PA0)
  // -------------------------------
  GPIO_InitStruct.Pin = Button0_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING; // Interrupt on rising edge
  GPIO_InitStruct.Pull = GPIO_PULLUP;         // Use pull-up resistor
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  // Enable and set EXTI line 0 interrupt priority
  HAL_NVIC_SetPriority(EXTI0_IRQn, 2, 0);
  HAL_NVIC_EnableIRQ(EXTI0_IRQn);

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */
/* USER CODE BEGIN 4 */
void EXTI0_IRQHandler(void){
	HAL_GPIO_EXTI_IRQHandler(Button0_Pin); // Clear interrupt flags

	uint32_t now = HAL_GetTick();

	if ((now - last_button_press) >= DEBOUNCE_DELAY)
	{
		last_button_press = now;
		// Stop DMA safely
		__HAL_TIM_DISABLE_DMA(&htim2, TIM_DMA_CC1);
		HAL_DMA_Abort_IT(&hdma_tim2_ch1);

		// TODO: Disable DMA transfer and abort IT, then start DMA in IT mode with new LUT and re-enable transfer
		current_waveform = (current_waveform + 1) % NUM_WAVEFORMS;

		switch (current_waveform)
		{
			case 0:
				lcd_command(CLEAR);
				lcd_putstring("SINE WAVE");
				HAL_DMA_Start_IT(&hdma_tim2_ch1,
								 (uint32_t)Sin_LUT,
								 (uint32_t)&(TIM3->CCR3),
								 NS);
				break;

			case 1:
				lcd_command(CLEAR);
				lcd_putstring("SAWTOOTH WAVE");
				HAL_DMA_Start_IT(&hdma_tim2_ch1,
								 (uint32_t)Saw_LUT,
								 (uint32_t)&(TIM3->CCR3),
								 NS);
				break;

			case 2:
				lcd_command(CLEAR);
				lcd_putstring("TRIANGLE WAVE");
				HAL_DMA_Start_IT(&hdma_tim2_ch1,
								 (uint32_t)Triangle_LUT,
								 (uint32_t)&(TIM3->CCR3),
								 NS);
				break;

			case 3:
				lcd_command(CLEAR);
				lcd_putstring("PIANO WAVE");
				HAL_DMA_Start_IT(&hdma_tim2_ch1,
								 (uint32_t)Triangle_LUT,
								 (uint32_t)&(TIM3->CCR3),
								 1024);
		   break;


			case 4:
				lcd_command(CLEAR);
				lcd_putstring("DRUM WAVE");
				HAL_DMA_Start_IT(&hdma_tim2_ch1,
								 (uint32_t)Triangle_LUT,
								 (uint32_t)&(TIM3->CCR3),
								 1024);
					break;

			case 5:
				lcd_command(CLEAR);
				lcd_putstring("GUITAR WAVE");
				HAL_DMA_Start_IT(&hdma_tim2_ch1,
								 (uint32_t)Triangle_LUT,
								 (uint32_t)&(TIM3->CCR3),
								 1024);
					break;

			// Optional: placeholder for future waveforms (e.g. piano, drum, guitar)
			default:
				lcd_command(CLEAR);
				lcd_putstring("SINE WAVE");
				HAL_DMA_Start_IT(&hdma_tim2_ch1,
								 (uint32_t)Sin_LUT,
								 (uint32_t)&(TIM3->CCR3),
								 1024);
				break;
		}
		// Re-enable DMA
		__HAL_TIM_ENABLE_DMA(&htim2, TIM_DMA_CC1);

	}
}
/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */