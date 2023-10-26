import numpy as np
import Enviroment as env
import matplotlib
import matplotlib.pyplot as plt
I=3
NUM_OF_DEVICE = 3
AP_POSITION=(45,45)
def distance_to_AP(pos_of_device):
    distance = np.sqrt((pos_of_device[0] - AP_POSITION[0]) * (pos_of_device[0] - AP_POSITION[0]) + (pos_of_device[1] - AP_POSITION[1]) * (pos_of_device[1] - AP_POSITION[1]))
    return distance
def path_loss_sub(distance):
    return 38.5 + 30*(np.log10(distance))
def compute_h_sub(list_of_devices,device_index,h_tilde):
    h=h_tilde* pow(10,-path_loss_sub(distance_to_AP(list_of_devices[device_index]))/20.0)
    return np.power(np.abs(h),2)

           
my_dict ={"java":100}
print((*my_dict,)[0])