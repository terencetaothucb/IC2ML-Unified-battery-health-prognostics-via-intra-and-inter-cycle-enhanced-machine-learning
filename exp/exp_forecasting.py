# from data_loader import *
from exp.exp_basic import Exp_Basic
from dataloader import *
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.tools import EarlyStopping, adjust_learning_rate, visual
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib.colors import LinearSegmentedColormap
import time
class Exp_Long_Term_Forecast1(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast1, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        return model
    def _get_data(self):
        if self.args.dataset == 'NCA':
            train_loader, X_test_tensor, y_test_tensor = NCA_trainloader(self.args)
        elif self.args.dataset == 'NCM':
            train_loader, X_test_tensor, y_test_tensor = NCM_trainloader(self.args)
        elif self.args.dataset == 'NCMNCA':
            train_loader, X_test_tensor, y_test_tensor = NCMNCA_trainloader(self.args)

        return train_loader, X_test_tensor, y_test_tensor

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def train(self, setting):
        train_loader, val_loader, test_loader = self._get_data()

        time_now = time.time()
        path = os.path.join(self.args.checkpoints,self.args.model, self.args.dataset ,self.args.condition,setting)
        if not os.path.exists(path):
            os.makedirs(path)
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for inputs, targets in train_loader:
                capacity_increment, start_volts, end_volts = inputs
                future_capacities, his_capacities,rul_real = targets
                capacity_increment = capacity_increment.to(self.device)
                start_volts = start_volts.to(self.device)
                end_volts = end_volts.to(self.device)
                future_capacities = future_capacities.to(self.device)
                his_capacities = his_capacities.to(self.device)
                rul_real = rul_real.to(self.device)
                SOH,trajectory,rul= self.model(capacity_increment, start_volts, end_volts)
                loss1 = criterion(trajectory, future_capacities)
                loss2 = criterion(SOH, his_capacities)
                loss3 = criterion(rul, rul_real)
                total_loss = loss2 + loss1 + 0.5*loss3
                train_loss.append(total_loss.item())
                model_optim.zero_grad()
                total_loss.backward()
                model_optim.step()
            train_loss = np.average(train_loss)
            vali_loss = self.vali(train_loader, val_loader, criterion)
            test_loss = vali_loss
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model
    
    def vali(self, train_loader, val_loader, criterion):
        self.model.eval()  
        
        val_loss = []  
        with torch.no_grad():  
            for inputs, targets in val_loader:
                capacity_increment, start_volts, end_volts = inputs
                future_capacities, his_capacities,rul_real = targets
                capacity_increment = capacity_increment.to(self.device)
                start_volts = start_volts.to(self.device)
                end_volts = end_volts.to(self.device)
                future_capacities = future_capacities.to(self.device)
                his_capacities = his_capacities.to(self.device)
                rul_real = rul_real.to(self.device)
                SOH,trajectory,rul= self.model(capacity_increment, start_volts, end_volts)
                loss1 = criterion(trajectory, future_capacities)
                loss2 = criterion(SOH, his_capacities)
                loss3 = criterion(rul, rul_real)
                total_loss = loss2 + loss1 + 0.5*loss3

                val_loss.append(total_loss.item()) 

        avg_val_loss = np.mean(val_loss)
        self.model.train()
        return avg_val_loss


    def test(self, setting, test=0):
        train_loader, val_loader, test_loader= self._get_data()
        self.model.eval()
        data_loader = test_loader
        total_rmse_fut = 0
        total_mape_fut = 0
        count = 0
        all_true_values_fut = []
        all_pred_values_fut = []
        total_rmse_his = 0
        total_mape_his = 0
        count = 0
        all_true_values_his = []
        all_pred_values_his = []
        total_rmse_rul = 0
        total_mape_rul = 0
        total_mae_rul = 0
        count = 0
        all_true_values_rul = []
        all_pred_values_rul = []
        all_true_rul = []
        all_pred_rul = []
        start_time = time.time()

        with torch.no_grad():
            for inputs, targets in data_loader:
                capacity_increment, start_volts, end_volts = inputs
                future_capacities, his_capacities,rul_real = targets

                capacity_increment = capacity_increment.to(self.device)
                start_volts = start_volts.to(self.device)
                end_volts = end_volts.to(self.device)

                future_capacities = future_capacities.to(self.device)
                his_capacities = his_capacities.to(self.device)
                rul_real = rul_real.to(self.device)

                his,future,rul= self.model(capacity_increment, start_volts, end_volts)


                y_pred_fut = future.cpu().numpy().reshape(future.shape)
                y_batch_fut = future_capacities.cpu().numpy().reshape(future_capacities.shape)

                rmse_fut = np.sqrt(np.mean((y_pred_fut - y_batch_fut)**2))
                mape_fut = np.mean(np.abs((y_pred_fut - y_batch_fut) / y_batch_fut)) * 100

                total_rmse_fut += rmse_fut 
                total_mape_fut += mape_fut 
                count += 1
                all_true_values_fut.append(y_batch_fut)
                all_pred_values_fut.append(y_pred_fut)

                y_pred_his = his.cpu().numpy().reshape(his.shape)
                y_batch_his = his_capacities.cpu().numpy().reshape(his_capacities.shape)

                rmse_his = np.sqrt(np.mean((y_pred_his - y_batch_his)**2))
                mape_his = np.mean(np.abs((y_pred_his - y_batch_his) / y_batch_his)) * 100

                total_rmse_his += rmse_his
                total_mape_his += mape_his
                count += 1
                all_true_values_his.append(y_batch_his)
                all_pred_values_his.append(y_pred_his)

                y_pred_rul = rul.cpu().numpy().reshape(rul.shape)
                y_batch_rul = rul_real.cpu().numpy().reshape(rul_real.shape)
                all_true_rul.append(y_batch_rul)
                all_pred_rul.append(y_pred_rul) 
                rmse_rul = np.sqrt(np.mean((y_pred_rul*100 - y_batch_rul*100)**2))
                mape_rul = np.mean(np.abs((y_pred_rul*100 - y_batch_rul*100) / (y_batch_rul*100))) * 100
                mae_rul = np.mean(np.abs(y_pred_rul*100 - y_batch_rul*100))
                total_rmse_rul += rmse_rul
                total_mape_rul += mape_rul
                total_mae_rul += mae_rul
                count += 1
                all_true_values_rul.append(y_batch_rul*100)
                all_pred_values_rul.append(y_pred_rul*100)

        end_time = time.time()
        total_test_time = end_time - start_time
        print(f"Total test time: {total_test_time:.2f} seconds")
        avg_rmse_fut = total_rmse_fut / count
        avg_mape_fut = total_mape_fut / count

        avg_rmse_his = total_rmse_his / count
        avg_mape_his = total_mape_his / count

        avg_rmse_rul = total_rmse_rul / count
        avg_mape_rul = total_mape_rul / count
        avg_mae_rul = total_mae_rul / count

        print(f"Average MAPE_rul (Normalized): {avg_mape_rul:.4f}%")
        print(f"Average MAE_rul (Normalized): {avg_mae_rul:.4f}")
        print(f"Average Test RMSE_rul: {avg_rmse_rul:.4f}")
        print(f"Average MAPE_fut (Normalized): {avg_mape_fut:.4f}%")
        print(f"Average Test RMSE_fut: {avg_rmse_fut:.4f}")
        print(f"Average MAPE_his (Normalized): {avg_mape_his:.4f}%")
        print(f"Average Test RMSE_his: {avg_rmse_his:.4f}")

        all_true_values_fut = np.concatenate(all_true_values_fut, axis=0)
        all_pred_values_fut = np.concatenate(all_pred_values_fut, axis=0)

        all_true_values_his = np.concatenate(all_true_values_his, axis=0)
        all_pred_values_his = np.concatenate(all_pred_values_his, axis=0)

        all_true_values_rul = np.concatenate(all_true_values_rul, axis=0)
        all_pred_values_rul = np.concatenate(all_pred_values_rul, axis=0)

        output_dir = os.path.join(self.args.checkpoints, self.args.model, self.args.dataset, self.args.condition,setting)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        np.save(os.path.join(output_dir, 'true_fut.npy'), all_true_values_fut)
        np.save(os.path.join(output_dir, 'pred_fut.npy'), all_pred_values_fut)

        np.save(os.path.join(output_dir, 'true_his.npy'), all_true_values_his)
        np.save(os.path.join(output_dir, 'pred_his.npy'), all_pred_values_his)

        np.save(os.path.join(output_dir, 'true_rul.npy'), all_true_values_rul)
        np.save(os.path.join(output_dir, 'pred_rul.npy'), all_pred_values_rul)
        plt.figure(figsize=(15, 8))
        for i in range(len(all_true_values_fut)):
            plt.plot(range(i, i + len(all_true_values_fut[i])), all_true_values_fut[i], color='blue', alpha=0.5, label='True Values' if i == 0 else "")
        for i in range(len(all_pred_values_fut)):
            plt.plot(range(i, i + len(all_pred_values_fut[i])), all_pred_values_fut[i], color='red', alpha=0.5, label='Predictions' if i == 0 else "")
        plt.legend()
        plt.title('All Samples: True Values vs Predictions_fut')
        plt.xlabel('Cycle Index')
        plt.ylabel('Discharge Capacity')
        plt.savefig(os.path.join(output_dir, 'fut.png')) 
        plt.close()
        plt.figure(figsize=(15, 8))
        for i in range(len(all_true_values_his)):
            plt.plot(range(i, i + len(all_true_values_his[i])), all_true_values_his[i], color='blue', alpha=0.5, label='True Values' if i == 0 else "")
        for i in range(len(all_pred_values_his)):
            plt.plot(range(i, i + len(all_pred_values_his[i])), all_pred_values_his[i], color='red', alpha=0.5, label='Predictions' if i == 0 else "")
        plt.legend()
        plt.title('All Samples: True Values vs Predictions_his')
        plt.xlabel('Cycle Index')
        plt.ylabel('Discharge Capacity')
        plt.savefig(os.path.join(output_dir, 'his.png')) 
        plt.close()
        all_true_rul = np.concatenate(all_true_rul, axis=0)
        all_pred_rul = np.concatenate(all_pred_rul, axis=0)

        plt.figure(figsize=(10, 8))
        plt.scatter(all_true_rul*100, all_pred_rul*100, alpha=0.6)
        plt.plot([min(all_true_rul*100), max(all_true_rul*100)], 
                [min(all_true_rul*100), max(all_true_rul*100)], 
                'k--', lw=2) 

        plt.xlabel('True RUL (cycles)')
        plt.ylabel('Predicted RUL (cycles)')
        plt.title('RUL Prediction: True vs Predicted')
        plt.savefig(os.path.join(output_dir, 'rul.png')) 


        return avg_rmse_fut, avg_mape_fut
