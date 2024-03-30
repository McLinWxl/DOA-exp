# DOA-exp

## File structure

```plaintext
DOA-exp
|-- Data
    |-- ULA_0.06
        |-- S1000
            |-- -a_b.npy
            |-- ...
        |-- S...
        |-- I2535
            |-- 0_-a_1_b.npy
            |-- ...
        |-- I...
        |-- Complete
            |-- H_-a_F_b.npy
            |-- ...
        |-- Inner
            |-- ...
        |-- Outer
            |-- ...
        |-- Ball
            |-- ...
    |-- ULA_0.03
        |-- ...
|-- DataProcess
    |-- DataProcess.py
    |-- __init__.py
    |-- __main__.py
    |-- functions.py
|-- DOA
    |-- DOA.py
    |-- __init__.py
    |-- __main__.py
    |-- functions.py
    |-- utils
        |-- __init__.py
        |-- dataset.py
        |-- model.py
|-- Test
|-- README.md
```
Sixteen antennas are used in the ULA.

Folder name format: 
1. Mian Folder: 'ULA_0.06' means the distance between two adjacent antennas is 0.06 wavelength.
2. Sub Folder:

| Case | Sub Folder Name                             | Description                                                                                                        |
|------|---------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| 1    | S1000/S1500/S1800 /S2000/S2500/S2800 /S5666 | Sinusoidal signal with different frequency. (S5666 have 5 minuets, others 20 seconds)                              |
| 2    | I2535/I1417                                 | Impulse signal with different center frequency. (Two sources have different fault characteristic frequency)        |
| 3    | Complete/Inner/Outer/Ball                   | Overall data, and inner, outer, ball data from 624 bearing dataset. (One source is health, and the other is fault) |

File name format: 

| Case | File Name   | Description                                                   |
|------|-------------|---------------------------------------------------------------|
| 1    | -a_b.h5     | Two source from -a and b degree.                              |
| 2    | 0_-a_1_b.h5 | Source 0 from -a degree, and source 1 from b degree.          |
| 3    | H_-a_F_b.h5 | Health source from -a degree, and fault source from b degree. |

**Notice**: 
1. For case 1, two sources are the same.
2. For case 2, two sources are different. Both have the same center frequency, but different fault characteristic frequency (Source 0 is 107 Hz, Source 1 is 87 Hz).
3. For case 3, two sources are different. One is health, and the other is fault.

