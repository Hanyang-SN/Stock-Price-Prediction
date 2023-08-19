import pandas as pd
import torch


class CustomDataset(torch.utils.data.Dataset):
	
	def __init__(self, df, input_size, output_size):
		self.x_data = []
		self.y_data = []
		
		last_idx = len(df) - input_size - output_size
		for r in range(last_idx):
			self.x_data.append(list(df.iloc[r:r + input_size]))
			self.y_data.append(list(df.iloc[r + input_size:r + input_size + output_size]))
	
	def __len__(self):
		return len(self.x_data)
	
	def __getitem__(self, idx):
		x = torch.FloatTensor(self.x_data[idx])
		y = torch.FloatTensor(self.y_data[idx])
		return x, y