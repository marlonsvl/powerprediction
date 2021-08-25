import matplotlib.pyplot as plt
import base64
from io import BytesIO
import numpy as np

def get_graph():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph

def get_plot(df):
	plt.switch_backend('AGG')
	df.plot(legend=True,figsize=(12,3))
	graph = get_graph()
	return graph

def get_bar(x, val, w, l, v1, l1, multi_performance):
	plt.switch_backend('AGG')
	plt.bar(x, val, w, label=l)
	plt.bar(x, v1, w, label=l1)
	plt.xticks(ticks=x, labels=multi_performance.keys(),rotation=45)
	plt.ylabel(f'MAE (average over all times and outputs)')
	graph = get_graph()
	return graph

def get_determination(x, val_mae, width, test_mae, multi_performance):
	plt.switch_backend('AGG')
	plt.figure(figsize=(12,5))
	plt.bar(x - 0.17, val_mae, width, label='Validation')
	plt.bar(x + 0.17, test_mae, width, label='Test')
	plt.xticks(ticks=x, labels=multi_performance.keys(), rotation=45)
	plt.ylabel(f'R2 (average over all times and outputs)')
	graph = get_graph()
	return graph	

def get_real_neural_normed(hybrid_pred, trues):
	plt.switch_backend('AGG')
	fig= plt.figure(figsize=(12,5))
	plt.title('Real Data vs. Neural Network Model Predictions [Normed]')
	plt.xlabel('Power Generation Day [Hour]')
	plt.ylabel('Power Generation [Normed]')
	plt.plot(hybrid_pred, label = 'Hybrid', marker = '<')
	plt.plot(trues, label = 'True', color = 'black', marker = '*', linewidth=3)
	plt.legend()
	plt.grid()
	graph = get_graph()
	return graph

def get_real_neural_kw(hybrid_pred, trues, train_std, train_mean):
	plt.switch_backend('AGG')
	fig= plt.figure(figsize=(12,5))
	plt.title('Real Data vs. Neural Network Model Predictions [kW]')
	plt.xlabel('Power Generation Day [Hour]')
	plt.ylabel('Power Generation [kW]')
	plt.plot(hybrid_pred*train_std[0]+train_mean[0], label = 'Hybrid', marker = '<')
	plt.plot(trues*train_std[0]+train_mean[0], label = 'True', color = 'black', marker = '*', linewidth=3)
	plt.legend()
	plt.grid()
	graph = get_graph()
	return graph


"""
def plot(self, model=None, plot_col='POWER', max_subplots=3):
	plt.switch_backend('AGG')
	pred = []
	inputs, labels = self.example
	plt.figure(figsize=(12, 3))
	plot_col_index = self.column_indices[plot_col]
	max_n = min(max_subplots, len(inputs))
	for n in range(max_n):
		plt.subplot(3, 1, n+1)
		plt.ylabel(f'{plot_col} [normed]')
		plt.plot(self.input_indices, inputs[n, :, plot_col_index],
         label='Inputs', marker='.', zorder=-10)
		if self.label_columns:
			label_col_index = self.label_columns_indices.get(plot_col, None)
		else:
			label_col_index = plot_col_index
		if label_col_index is None:
			continue
		trues = labels[n, :, label_col_index].numpy()
    	#plt.scatter(self.label_indices, labels[n, :, label_col_index], edgecolors='k', label='Labels', c='#2ca02c', s=64)
    	if model is not None:
    		predictions = model(inputs)
    		plt.scatter(self.label_indices, predictions[n, :, label_col_index], marker='X', edgecolors='k', label='Predictions', c='#ff7f0e', s=64)
    		pred = predictions[n, :, label_col_index].numpy()  
    	if n == 0:
    		plt.legend()

  	plt.xlabel('Time [h]')
  	graph = get_graph()
  	return graph

"""