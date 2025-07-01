# physical constants:
g = 9.80665 # m/s^2
mps_to_mph = 2.23694 # m/s to mph
mile_to_km = 1.60934 # km/mile
mile_to_ft = 5280 # ft/mile
mile_to_m = 1609.34 # m/mile
ft_to_m = 0.3048 # m/ft

# make_dataset_sl:

sl_file_names = [
    'Location',
    'Orientation',
    # 'GyroscopeUncalibrated', # eliminated
    'Accelerometer',
    'AccelerometerUncalibrated',
    'Gyroscope',
    'Barometer',
] # deal with Metadata later; exclude for now

sl_cols_maps = {
    'Magnetometer': {'x': 'mx_device', 'y': 'my_device', 'z': 'mz_device'},
    'GyroscopeUncalibrated':  {'x': 'gx_uc_device', 'y': 'gy_uc_device', 'z': 'gz_uc_device'},
    'MagnetometerUncalibrated':  {'x': 'mx_uc_device', 'y': 'my_uc_device', 'z': 'mz_uc_device'},
    'Accelerometer':  {'x': 'ax_device', 'y': 'ay_device', 'z': 'az_device'},
    'AccelerometerUncalibrated': {'x': 'ax_uc_device', 'y': 'ay_uc_device', 'z': 'az_uc_device'},
    'Gyroscope': {'x': 'gx_device', 'y': 'gy_device', 'z': 'gz_device'},
    'Gravity': {'x': 'gFx_device', 'y': 'gFy_device', 'z': 'gFz_device'},
}

interp_cols = [
    'altitude',
    'altitudeAboveMeanSeaLevel',
    'bearing',
    'bearingAccuracy',
    'horizontalAccuracy',
    'verticalAccuracy',
    'latitude',
    'longitude',
    'speed',
    'speedAccuracy',
    'relativeAltitude',
    'pressure',
]


# features and transforms:
magnitude_mapper = {
    ('ax_device', 'ay_device', 'az_device'): 'ar',
    ('ax_uc_device', 'ay_uc_device', 'az_uc_device'): 'ar_uc',
    ('gx_device', 'gy_device', 'gz_device'): 'gr',
}

earth_to_travel_mapper = {
    ('ax_e', 'ay_e', 'az_e'): ('ax_t', 'ay_t', 'az_t'),
    ('gx_e', 'gy_e', 'gz_e'): ('gx_t', 'gy_t', 'gz_t'),
    ('ax_uc_e', 'ay_uc_e', 'az_uc_e'): ('ax_uc_t', 'ay_uc_t', 'az_uc_t'),
}

lead_lag_cols = ['ar', 'gr']

'''
lead_lag_cols = [
    'ar', 'gr', 'ar_uc', 
    'az_e', 'az_uc_e', 'gz_e',
    'gx_e', 'gy_e', 'gz_e',
]
'''

# jumps:
jump_z_accel_threshold = 1.5
jump_min_consecutive_points = 2
