# Preprocessing data for machine learning applications
Due to the large size of the database, it cannot be entirely loaded on CPUs or GPUs. Therefore, the preprocessing step consists in writing individual files `sample_i.h5` that contain the material `a` and the three components of the velocity fields `uE`, `uN`, and `uZ`. 

To reduce the computational time of machine learning applications, velocity fields are downsampled from 100 Hz to 50 Hz, and restricted to the time interval [1; 7.4s] (leading to 320 time steps). They are also spatially interpolated from 16 x 16 sensors to 32 x 32 to match the inputs dimension.

To create the inputs, run `python3 create_data_materials.py @Ntrain 27000 @Nval 3000` and then `python3 create_data_velocityfields.py @Ntrain 27000 @Nval 3000 @interpolate`. This takes around 1.5 hour. 

Outputs of these codes are saved in models/inputs. 

To train the ML models, the mean and standard deviation of materials are also needed (to normalize inputs). They can be created easily by loading all materials in data and computing the mean and std along axis 0. 

