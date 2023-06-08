from datetime import datetime
import torch

def save_model(model, time):
    torch.save(model.state_dict(), "auto_iv_"+time+".pth")

class Log():
    def __init__(self, log_path):
        self.time = datetime.now().strftime("%Y-%m-%d-%H-%M")

        self.log_path = log_path
        self.log_file = log_path + self.time + ".log"
    
    def write(self, _str):
        with open(self.log_file, 'a') as f:
            f.write(_str + "\n")