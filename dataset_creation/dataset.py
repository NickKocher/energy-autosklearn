import numpy as np
import pandas as pd
from os.path import exists

from bisect import bisect_left
ONEMILLION = 1000000
#Code adapted from https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value/12141511#12141511
def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns index from closest value to myNumber.

    If two numbers are equally close, return index from the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return 0
    if pos == len(myList):
        return len(myList)-1
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return pos
    else:
        return pos-1

def postprocess_energy_data(raw_measurements:pd.DataFrame):
    drop_indices = [x for x in range(raw_measurements.shape[0]) if raw_measurements.at[x,'energy'] == 0]
    print(raw_measurements.shape[0])
    energy_measurements = raw_measurements.drop(index=drop_indices)
    #energy_measurements = energy_measurements.drop(columns=['name','channel','current','power','voltage'],inplace=False)
    print(energy_measurements.iloc[0])
    timing_list = energy_measurements.iloc[:,1].to_list()
    #print(timing_list)
    return energy_measurements,timing_list

def get_nearest_energy_estimation(
    start: int, duration: int, energy_measurements: pd.DataFrame,timing_list:list
):
    #create list from dataframe
    
    start_index = take_closest(timing_list,start)
    end_index = take_closest(timing_list,start+duration)
    #print(f"{start_index},{end_index}")
    #print(energy_measurements.shape)
    energy = energy_measurements.iloc[end_index,4] - energy_measurements.iloc[start_index,4]
    #calculate offsets in ms
    start_offset = np.abs(start-energy_measurements.iloc[start_index,1])//ONEMILLION
    end_offset = np.abs(start+duration-energy_measurements.iloc[end_index,1])//ONEMILLION
    return energy,start_offset,end_offset


# assumes config measurements are in ascending order, e.g /path/to/measurement_0 /path/to/measurement_1
def associate_config_energy(
    path_to_first_config_measurement: str, path_to_energy_measurements: str
):
    df = pd.read_csv(path_to_energy_measurements)
    config_path = path_to_first_config_measurement[:-1]
    print(config_path)
    energy_measurements,timing_list = postprocess_energy_data(raw_measurements=df)
    index = 0
    output_dataframe = pd.DataFrame(data=[],columns=['algorithm','time','energy','inaccuracy_start','inaccuracy_end'])
    #print(''.join([config_path,f"{index}"]))
    while exists(''.join([config_path,f"{index}"])):
        current_path = ''.join([config_path,f"{index}"])
        current_configuration_data = np.load(current_path, allow_pickle=True)
        current_energy,start_offset,end_offset = get_nearest_energy_estimation(
            start=current_configuration_data[0],
            duration=current_configuration_data[1],
            energy_measurements=energy_measurements,
            timing_list=timing_list
        )
        if current_energy < 0:
            break
        #output_dataframe.loc[len(df.index)] = [current_configuration_data[2]["classifier:__choice__"],current_energy,current_configuration_data[1],start_offset,end_offset]
        print(f"{[current_configuration_data[2]['classifier:__choice__'],current_energy,current_configuration_data[1]//ONEMILLION]}")
        #print(f"{current_energy},")
        index = index + 1
    return 0


def main():
    associate_config_energy("/home/kocher/energy-autosklearn/metadata-retraining/run_/1710692225784161602_0","/home/kocher/energy-autosklearn/metadata-retraining/test_run.csv")


if __name__ == "__main__":
    main()
