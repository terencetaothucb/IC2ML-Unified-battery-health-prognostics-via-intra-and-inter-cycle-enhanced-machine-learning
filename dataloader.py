import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis
from torch.utils.data import DataLoader, Dataset
import random
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class BatteryDataset(Dataset):
    def __init__(self, data, context=10, horizon=50, capacity_length=10, scaler_features=None):
        min_discharge_capacity = data['Discharge_Capacity'].min()
        if min_discharge_capacity > 2850:
            raise ValueError(f"min_discharge_capacity{min_discharge_capacity}")
        self.data = data
        self.context = context
        self.horizon = horizon
        self.capacity_length = capacity_length
        self.capacity_increment_list = []
        self.start_voltages = []
        self.end_voltages = []
        self.rul_values = []
        threshold = 2800 
        total_cycles = len(data)
        end_of_life_idx = None
        for i in range(len(data)):
            if data.iloc[i]['Discharge_Capacity'] < threshold:
                end_of_life_idx = i
                break
        if end_of_life_idx is None:
            end_of_life_idx = total_cycles - 1
        for i in range(len(data)):
            current_rul = max(0, end_of_life_idx - i)  
            self.rul_values.append(current_rul)

        for i in range(len(data)):
            cap_incr = data.iloc[i]['Capacity_Increment']
            if isinstance(cap_incr, str):
                cap_incr = eval(cap_incr)
            self.capacity_increment_list.append(np.array(cap_incr) / 1000)
            
            self.start_voltages.append(data.iloc[i]['Start_Voltage'])
            self.end_voltages.append(data.iloc[i]['End_Voltage'])
              
        self.capacity_increment_list = np.array(self.capacity_increment_list) 
        self.start_voltages = np.array(self.start_voltages) 
        self.end_voltages = np.array(self.end_voltages)  
        self.rul_values = np.array(self.rul_values)  
    def __len__(self):
        return len(self.data) - self.context - self.horizon + 1

    def __getitem__(self, idx):
        history_increments = self.capacity_increment_list[idx:idx+self.context]  
        start_volts = self.start_voltages[idx:idx+self.context] 
        end_volts = self.end_voltages[idx:idx+self.context]  
        increment_tensor = torch.tensor(history_increments, dtype=torch.float32)
        start_volts = torch.tensor(start_volts, dtype=torch.float32)
        end_volts = torch.tensor(end_volts, dtype=torch.float32)
        inputs = (increment_tensor, start_volts, end_volts)
        his_capacities = self.data.iloc[
            idx:idx+self.context
        ]['Discharge_Capacity'].values / 1000 
        future_capacities = self.data.iloc[
            idx+self.context : idx+self.context+self.horizon
        ]['Discharge_Capacity'].values / 1000  
        current_cycle = idx + self.context - 1 
        current_rul = self.rul_values[current_cycle]  
        future_capacities = torch.tensor(future_capacities, dtype=torch.float32)
        his_capacities = torch.tensor(his_capacities, dtype=torch.float32)
        current_rul = torch.tensor(current_rul, dtype=torch.float32)
        outputs = (future_capacities, his_capacities, current_rul/100)
        return inputs, outputs
    
class BatteryDataset1(Dataset):
    def __init__(self, data, context=10, horizon=50, capacity_length=10, scaler_features=None):
        min_discharge_capacity = data['Discharge_Capacity'].min()
        if min_discharge_capacity > 2050:
            raise ValueError(f"min_discharge_capacity{min_discharge_capacity}")
        self.data = data
        self.context = context
        self.horizon = horizon
        self.capacity_length = capacity_length
        self.capacity_increment_list = []
        self.start_voltages = []
        self.end_voltages = []
        self.rul_values = []
        threshold = 2000 
        total_cycles = len(data)
        end_of_life_idx = None
        for i in range(len(data)):
            if data.iloc[i]['Discharge_Capacity'] < threshold:
                end_of_life_idx = i
                break
        if end_of_life_idx is None:
            end_of_life_idx = total_cycles - 1
        for i in range(len(data)):
            current_rul = max(0, end_of_life_idx - i)  
            self.rul_values.append(current_rul)

        for i in range(len(data)):
            cap_incr = data.iloc[i]['Capacity_Increment']
            if isinstance(cap_incr, str):
                cap_incr = eval(cap_incr)
            self.capacity_increment_list.append(np.array(cap_incr) / 1000)
            self.start_voltages.append(data.iloc[i]['Start_Voltage'])
            self.end_voltages.append(data.iloc[i]['End_Voltage'])
              
        self.capacity_increment_list = np.array(self.capacity_increment_list) 
        self.start_voltages = np.array(self.start_voltages) 
        self.end_voltages = np.array(self.end_voltages)  
        self.rul_values = np.array(self.rul_values)  
    def __len__(self):
        return len(self.data) - self.context - self.horizon + 1

    def __getitem__(self, idx):
        history_increments = self.capacity_increment_list[idx:idx+self.context]  
        start_volts = self.start_voltages[idx:idx+self.context] 
        end_volts = self.end_voltages[idx:idx+self.context]  
        increment_tensor = torch.tensor(history_increments, dtype=torch.float32)
        start_volts = torch.tensor(start_volts, dtype=torch.float32)
        end_volts = torch.tensor(end_volts, dtype=torch.float32)
        inputs = (increment_tensor, start_volts, end_volts)
        his_capacities = self.data.iloc[
            idx:idx+self.context
        ]['Discharge_Capacity'].values / 1000 
        future_capacities = self.data.iloc[
            idx+self.context : idx+self.context+self.horizon
        ]['Discharge_Capacity'].values / 1000  
        current_cycle = idx + self.context - 1 
        current_rul = self.rul_values[current_cycle]  
        future_capacities = torch.tensor(future_capacities, dtype=torch.float32)
        his_capacities = torch.tensor(his_capacities, dtype=torch.float32)
        current_rul = torch.tensor(current_rul, dtype=torch.float32)
        outputs = (future_capacities, his_capacities, current_rul/100)
        return inputs, outputs


def NCA_trainloader(args):
    train_samples = []
    train_features_list = []
    train_outputs = []
    val_samples = []
    val_outputs = []
    test_samples = []
    test_outputs = []
    if args.condition == 'CY45-05_1':
        train_files = [
            'CY45-05_1-#1.csv', 'CY45-05_1-#2.csv', 'CY45-05_1-#3.csv', 'CY45-05_1-#4.csv',
            'CY45-05_1-#5.csv', 'CY45-05_1-#6.csv', 'CY45-05_1-#7.csv', 'CY45-05_1-#8.csv',
            'CY45-05_1-#9.csv', 'CY45-05_1-#10.csv', 'CY45-05_1-#11.csv', 'CY45-05_1-#12.csv',
            'CY45-05_1-#13.csv', 'CY45-05_1-#14.csv', 'CY45-05_1-#15.csv', 'CY45-05_1-#16.csv',
            'CY45-05_1-#17.csv'
        ]
        val_files = [
            'CY45-05_1-#28.csv', 'CY45-05_1-#25.csv'
        ]
        test_files = [
            'CY45-05_1-#24.csv', 'CY45-05_1-#26.csv', 'CY45-05_1-#27.csv', 'CY45-05_1-#22.csv',
            'CY45-05_1-#23.csv'
        ]
    elif args.condition == 'CY25-05_1':
        train_files = [
            'CY25-05_1-#2.csv', 'CY25-05_1-#3.csv', 'CY25-05_1-#4.csv','CY25-05_1-#18.csv',
            'CY25-05_1-#5.csv', 'CY25-05_1-#6.csv', 'CY25-05_1-#7.csv', 'CY25-05_1-#8.csv',
            'CY25-05_1-#9.csv', 'CY25-05_1-#10.csv', 'CY25-05_1-#11.csv', 'CY25-05_1-#13.csv'
        ]
        val_files = [
            'CY25-05_1-#18.csv', 'CY25-05_1-#19.csv'
        ]
        test_files = [
             'CY25-05_1-#1.csv', 'CY25-05_1-#14.csv', 'CY25-05_1-#15.csv', 'CY25-05_1-#16.csv',
            'CY25-05_1-#17.csv', 'CY25-05_1-#12.csv'
        ]
    elif args.condition == 'CY25-025_1':
        train_files = [
            'CY25-025_1-#1.csv', 'CY25-025_1-#2.csv', 'CY25-025_1-#3.csv'
        ]
        val_files = [
            'CY25-025_1-#7.csv'
        ]
        test_files = [
            'CY25-025_1-#5.csv', 'CY25-025_1-#6.csv', 'CY25-025_1-#4.csv'
        ]
    elif args.condition == 'CY25-1_1':
        train_files = [
            'CY25-1_1-#1.csv', 'CY25-1_1-#2.csv', 'CY25-1_1-#3.csv', 'CY25-1_1-#4.csv', 'CY25-1_1-#5.csv'
        ]
        val_files = [
            'CY25-1_1-#6.csv'
        ]
        test_files = [
            'CY25-1_1-#7.csv', 'CY25-1_1-#8.csv', 'CY25-1_1-#9.csv'
        ]
    elif args.condition == 'CY35-05_1':
        train_files = [
            'CY35-05_1-#1.csv'
        ]
        val_files = [
            'CY35-05_1-#2.csv'
        ]
        test_files = [
            'CY35-05_1-#3.csv'
        ]
    else:
        raise ValueError(f"Unsupported condition: {args.condition}")
    

    
    input_folders = [
        'dataset/NCA/V3.6-3.7/',
    ]

    if hasattr(args, 'dataaccess'):
        if args.dataaccess == 100:
            train_files = train_files.copy()
        else:
            num_train = max(1, math.ceil(len(train_files) * args.dataaccess / 100))
            train_files = random.sample(train_files, num_train)
    else:
        train_files = train_files.copy()

    for input_folder in input_folders:
        for file_name in train_files:
            file_path = os.path.join(input_folder, file_name)
            if not os.path.exists(file_path):
                continue  
                
            data = pd.read_csv(file_path)
            data['Capacity_Increment'] = data['Capacity_Increment'].apply(lambda x: eval(x))
            try:
                battery_dataset = BatteryDataset(data, context=args.context, horizon=args.horizon)
            except ValueError as e:
                print(f" {file_path}：{e}")
                continue
            
            for idx in range(len(battery_dataset)):
                inputs, outputs = battery_dataset[idx]
                train_samples.append(inputs)

                train_outputs.append(outputs)
    for input_folder in input_folders:  
        for file_name in val_files:
            file_path = os.path.join(input_folder, file_name)
            if not os.path.exists(file_path):
                continue
                
            data = pd.read_csv(file_path)
            data['Capacity_Increment'] = data['Capacity_Increment'].apply(lambda x: eval(x))
            try:
                battery_dataset = BatteryDataset(data, context=args.context, horizon=args.horizon)
            except ValueError as e:
                print(f" {file_path}：{e}")
                continue
            
            for idx in range(len(battery_dataset)):
                inputs, outputs = battery_dataset[idx]
                val_samples.append(inputs)
                val_outputs.append(outputs)

    for input_folder in input_folders:
        for file_name in test_files:
            file_path = os.path.join(input_folder, file_name)
            if not os.path.exists(file_path):
                continue
                
            data = pd.read_csv(file_path)
            data['Capacity_Increment'] = data['Capacity_Increment'].apply(lambda x: eval(x))
            try:
                battery_dataset = BatteryDataset(data, context=args.context, horizon=args.horizon)
            except ValueError as e:
                print(f" {file_path}：{e}")
                continue
            
            for idx in range(len(battery_dataset)):
                inputs, outputs = battery_dataset[idx]
                test_samples.append(inputs)
                test_outputs.append(outputs)

    train_loader = DataLoader(
        list(zip(train_samples, train_outputs)),
        batch_size=args.batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        list(zip(val_samples, val_outputs)),
        batch_size=args.batch_size, 
        shuffle=False  
    )
    test_loader = DataLoader(
        list(zip(test_samples, test_outputs)),
        batch_size=args.batch_size, 
        shuffle=False
    )

    return train_loader, val_loader, test_loader

def NCM_trainloader(args):
    train_samples = []
    train_features_list = []
    train_outputs = []
    val_samples = []
    val_outputs = []
    test_samples = []
    test_outputs = []

    if args.condition == 'CY25-05_1':
        train_files = [
            'CY25-05_1-#1.csv', 'CY25-05_1-#2.csv', 'CY25-05_1-#3.csv', 'CY25-05_1-#4.csv',
            'CY25-05_1-#5.csv', 'CY25-05_1-#6.csv', 'CY25-05_1-#7.csv', 'CY25-05_1-#8.csv',
            'CY25-05_1-#9.csv', 'CY25-05_1-#10.csv', 'CY25-05_1-#11.csv', 'CY25-05_1-#12.csv',
            'CY25-05_1-#8.csv'
        ]
        val_files = [
            'CY25-05_1-#5.csv','CY25-05_1-#17.csv', 'CY25-05_1-#15.csv', 'CY25-05_1-#16.csv'
        ]
        test_files = [
            'CY25-05_1-#13.csv', 'CY25-05_1-#19.csv', 'CY25-05_1-#20.csv', 'CY25-05_1-#21.csv',
            'CY25-05_1-#22.csv', 'CY25-05_1-#23.csv'
        ]

    elif args.condition == 'CY45-05_1':
        train_files = [
            'CY45-05_1-#1.csv', 'CY45-05_1-#2.csv', 'CY45-05_1-#3.csv', 'CY45-05_1-#4.csv',
            'CY45-05_1-#5.csv', 'CY45-05_1-#6.csv', 'CY45-05_1-#7.csv', 'CY45-05_1-#8.csv',
            'CY45-05_1-#9.csv', 'CY45-05_1-#10.csv', 'CY45-05_1-#11.csv', 'CY45-05_1-#12.csv',
            'CY45-05_1-#13.csv', 'CY45-05_1-#14.csv', 'CY45-05_1-#15.csv', 'CY45-05_1-#16.csv',

        ]
        val_files = [
            'CY45-05_1-#28.csv','CY45-05_1-#17.csv'
        ]
        test_files = [
            'CY45-05_1-#24.csv', 'CY45-05_1-#26.csv', 'CY45-05_1-#27.csv', 'CY45-05_1-#22.csv',
            'CY45-05_1-#23.csv'
        ]

    elif args.condition == 'CY35-05_1':
        train_files = [
            'CY35-05_1-#1.csv'
        ]
        val_files = [
            'CY35-05_1-#2.csv'
        ]
        test_files = [
            'CY35-05_1-#3.csv', 'CY35-05_1-#4.csv'
        ]
    else:
        raise ValueError(f"Unsupported condition: {args.condition}")
    input_folders = [
        'dataset/NCM/V3.6-3.7/',
    ]

    if hasattr(args, 'dataaccess'):
        if args.dataaccess == 100:
            train_files = train_files.copy()
        else:
            num_train = max(1, math.ceil(len(train_files) * args.dataaccess / 100))
            train_files = random.sample(train_files, num_train)
    else:
        train_files = train_files.copy()

    for input_folder in input_folders:
        for file_name in train_files:
            file_path = os.path.join(input_folder, file_name)
            if not os.path.exists(file_path):
                continue  
                
            data = pd.read_csv(file_path)
            data['Capacity_Increment'] = data['Capacity_Increment'].apply(lambda x: eval(x))
            try:
                battery_dataset = BatteryDataset(data, context=args.context, horizon=args.horizon)
            except ValueError as e:
                print(f" {file_path}：{e}")
                continue
            
            for idx in range(len(battery_dataset)):
                inputs, outputs = battery_dataset[idx]
                train_samples.append(inputs)

                train_outputs.append(outputs)

    for input_folder in input_folders:  
        for file_name in val_files:
            file_path = os.path.join(input_folder, file_name)
            if not os.path.exists(file_path):
                continue
                
            data = pd.read_csv(file_path)
            data['Capacity_Increment'] = data['Capacity_Increment'].apply(lambda x: eval(x))
            try:
                battery_dataset = BatteryDataset(data, context=args.context, horizon=args.horizon)
            except ValueError as e:
                print(f" {file_path}：{e}")
                continue
            
            for idx in range(len(battery_dataset)):
                inputs, outputs = battery_dataset[idx]
                val_samples.append(inputs)
                val_outputs.append(outputs)

    for input_folder in input_folders:
        for file_name in test_files:
            file_path = os.path.join(input_folder, file_name)
            if not os.path.exists(file_path):
                continue
            data = pd.read_csv(file_path)
            data['Capacity_Increment'] = data['Capacity_Increment'].apply(lambda x: eval(x))
            try:
                battery_dataset = BatteryDataset(data, context=args.context, horizon=args.horizon)
            except ValueError as e:
                print(f" {file_path}：{e}")
                continue
            
            for idx in range(len(battery_dataset)):
                inputs, outputs = battery_dataset[idx]
                test_samples.append(inputs)
                test_outputs.append(outputs)

    train_loader = DataLoader(
        list(zip(train_samples, train_outputs)),
        batch_size=args.batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        list(zip(val_samples, val_outputs)),
        batch_size=args.batch_size, 
        shuffle=False  
    )
    test_loader = DataLoader(
        list(zip(test_samples, test_outputs)),
        batch_size=args.batch_size, 
        shuffle=False
    )

    return train_loader, val_loader, test_loader

def NCMNCA_trainloader(args):
    train_samples = []
    train_features_list = []
    train_outputs = []
    val_samples = []
    val_outputs = []
    test_samples = []
    test_outputs = []
    if args.condition == 'CY25-05_1':
        train_files = [
            'CY25-05_1-#1.csv'
        ]
        val_files = [
            'CY25-05_1-#2.csv'
        ]
        test_files = [
            'CY25-05_1-#3.csv'
        ]

    elif args.condition == 'CY25-05_2':
        train_files = [
            'CY25-05_2-#1.csv'
        ]
        val_files = [
            'CY25-05_2-#2.csv'
        ]
        test_files = [
            'CY25-05_2-#3.csv'
        ]

    elif args.condition == 'CY25-05_4':
        train_files = [
            'CY25-05_4-#1.csv'
        ]
        val_files = [
            'CY25-05_4-#2.csv'
        ]
        test_files = [
            'CY25-05_4-#3.csv'
        ]
    else:
        raise ValueError(f"Unsupported condition: {args.condition}")
    input_folders = [
        'dataset/NCMNCA/V3.6-3.7/',

    ]

    if hasattr(args, 'dataaccess'):
        if args.dataaccess == 100:
            train_files = train_files.copy()
        else:
            num_train = max(1, math.ceil(len(train_files) * args.dataaccess / 100))
            train_files = random.sample(train_files, num_train)
    else:
        train_files = train_files.copy()

    for input_folder in input_folders:
        for file_name in train_files:
            file_path = os.path.join(input_folder, file_name)
            if not os.path.exists(file_path):
                continue 
                
            data = pd.read_csv(file_path)
            data['Capacity_Increment'] = data['Capacity_Increment'].apply(lambda x: eval(x))
            try:
                battery_dataset = BatteryDataset1(data, context=args.context, horizon=args.horizon)
            except ValueError as e:
                print(f" {file_path}：{e}")
                continue
            
            for idx in range(len(battery_dataset)):
                inputs, outputs = battery_dataset[idx]
                train_samples.append(inputs)

                train_outputs.append(outputs)

    for input_folder in input_folders: 
        for file_name in val_files:
            file_path = os.path.join(input_folder, file_name)
            if not os.path.exists(file_path):
                continue
            data = pd.read_csv(file_path)
            data['Capacity_Increment'] = data['Capacity_Increment'].apply(lambda x: eval(x))
            try:
                battery_dataset = BatteryDataset1(data, context=args.context, horizon=args.horizon)
            except ValueError as e:
                print(f" {file_path}：{e}")
                continue
            
            for idx in range(len(battery_dataset)):
                inputs, outputs = battery_dataset[idx]
                val_samples.append(inputs)
                val_outputs.append(outputs)

    for input_folder in input_folders:
        for file_name in test_files:
            file_path = os.path.join(input_folder, file_name)
            if not os.path.exists(file_path):
                continue
                
            data = pd.read_csv(file_path)
            data['Capacity_Increment'] = data['Capacity_Increment'].apply(lambda x: eval(x))
            try:
                battery_dataset = BatteryDataset1(data, context=args.context, horizon=args.horizon)
            except ValueError as e:
                print(f" {file_path}：{e}")
                continue
            
            for idx in range(len(battery_dataset)):
                inputs, outputs = battery_dataset[idx]
                test_samples.append(inputs)
                test_outputs.append(outputs)
    train_loader = DataLoader(
        list(zip(train_samples, train_outputs)),
        batch_size=args.batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        list(zip(val_samples, val_outputs)),
        batch_size=args.batch_size, 
        shuffle=False  
    )
    test_loader = DataLoader(
        list(zip(test_samples, test_outputs)),
        batch_size=args.batch_size, 
        shuffle=False
    )

    return train_loader, val_loader, test_loader