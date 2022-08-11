from data_prep import *
from net import *
import pickle
from datetime import datetime, timedelta
from pandas.tseries.frequencies import to_offset
import numpy as np
import pandas as pd
import PyPluMA

def main(data_path, data_covariates_path, date_column, save_path, prediction_name, model_config_path, meta_information_path, steps, flag_real_plot) :
	# model config :
	with open(model_config_path, 'r') as file:
		model_config = json.load(file)

	# Metainformation :
	with open(meta_information_path, 'r') as file:
		meta_information = json.load(file)

	# Covariates and Covariates_var
	with open(meta_information['transformation_covariates_var_path'], 'rb') as file:
		transformation_covariates_var = pickle.load(file)

	with open(meta_information['transformation_covariates'], 'rb') as file:
		transformation_covariates = pickle.load(file)

	flag_covariate = not (transformation_covariates is None)
	flag_covariate_var = not (transformation_covariates_var is None)

	if (flag_covariate_var == True ) and (data_covariates_path is None) :
		raise ValueError('Data have a covariates_var and no path to it is gived !')


	# Model Definition :
	window_size = model_config['seq_length']
	number_of_vars = model_config['k']
	num_timeseries_kinds = model_config['num_tokens']

	headers =  model_config['headers']
	depth = model_config['depth']
	kernel_size = model_config['kernel_size']
	PATH = model_config['path']

	train_end = meta_information['train_end']

	#############################################################################

	# Data loading

	print('## Data reading :')
	if date_column :
		Table = pd.read_csv(data_path)
		Table[date_column] = pd.to_datetime(Table[date_column])
		Table.set_index(date_column,inplace=True)
	else : 
		Table = pd.read_csv(data_path, index_col=0, parse_dates=True)
		inf_index = Table.index[np.isinf(Table).any(1)]
		inf_col =  Table.columns.to_series()[np.isinf(Table).any()]

	if data_covariates_path:
		print('## Data covariates reading :')
		covariates_var = []
		for file in os.listdir(data_covariates_path):
			if file.endswith(".csv"):
				if date_column :
					var = pd.read_csv(data_covariates_path+'/'+file)
					var[date_column] = pd.to_datetime(var[date_column])
					var.set_index(date_column, inplace=True)
				else :
					var = pd.read_csv(data_covariates_path+'/'+file, index_col = 0, parse_date=True)
				covariates_var.append(var)
	else :
		covariates_var = None

	print('# Preparing the data :')
	# Train and Test splitfrom pandas.tseries.frequencies import to_offset

	freq = pd.infer_freq(Table.index)
	freq = to_offset(freq)

	#predict starting 

	test_start = Table.index[0] 
	test_end = Table.index[-1]
	if flag_real_plot:
		test_end = Table.index[-2]
		assert steps == 1


	# List of time series
	timeseries_list = np.array(list(Table.columns))

	# covariate variables preparation
	test_data = Table[test_start:test_end].values
	tst_index = Table[test_start:test_end].index

	if not covariates_var is None : 
		test_covariates_var = []
		for var in covariates_var :
			assert (tst_index.union([tst_index[-1] + i*freq for i in range(steps+1)]) == var[test_start:test_end+steps*freq].index).all(), 'The covariate var '+str(var)+' didn t much steps for prediction : '+str(tst_index.union([tst_index[-1] + i*freq for i in range(steps+1)]))+' != '+str(var.index[test_start:test_end+steps*freq])
			test_covariates_var.append(var[test_start:test_end+steps*freq].values[:,:,None])
		test_covariates_var = np.concatenate(test_covariates_var,axis=2)
	else :
		test_covariates_var = None

	## Test

	freq = pd.infer_freq(tst_index)
	freq = to_offset(freq)

	if flag_covariate :
		covariates_test = gen_covariates(tst_index.union([tst_index[-1] + i*freq for i in range(steps+1)]))
	else :
		covariates_test = False

	# data_start
	data_start_test = (test_data!=0).argmax(axis=0) #find first nonzero value in each time series

	# Apply the transformation
	print('# Transforming the data :')
	window_size_test = window_size

	device = 'cpu'
	if torch.cuda.is_available():
		torch.set_default_tensor_type(torch.cuda.FloatTensor)
		device = 'cuda'

	###################

	# Model 
	model = ForcastConvTransformer(k = number_of_vars, headers=headers, depth=depth, seq_length=window_size, kernel_size=kernel_size, num_tokens=num_timeseries_kinds)
	#####################
	print(PATH)
	checkpoint = torch.load(PATH)
	model.load_state_dict(checkpoint['model_state_dict'])
	model.eval()

	estimation, sigma = prep_data_for_prediction(model, test_data, window_size_test, steps=steps, data_start=data_start_test, timeseries_list=timeseries_list,device = 'cuda', covariates=covariates_test, covariates_var=test_covariates_var,transformation_covariates_used = transformation_covariates, transformation_covariates_var_used = transformation_covariates_var, sampling = False)

	pt = os.path.join(save_path, prediction_name)
	os.makedirs(pt, exist_ok = True)

	np.save(pt+'/estimation.npy',estimation)
	np.save(pt+'/sigma.npy',sigma)


	# show a plot

	m= np.random.randint(0,estimation.shape[1])
	if flag_real_plot:
		plt.plot(Table[test_start:].iloc[4:,m].values,label='real', c='red')
	plt.plot(estimation[4:,m], label='prediction', c='b')

	under_line = estimation[4:,m] - sigma[4:,m]
	over_line = estimation[4:,m] + sigma[4:,m]
	plt.fill_between(range(len(estimation[4:,m])),under_line, over_line, color='b', alpha=0.2) #std curves.

	if train_end in Table.index :
		plt.axvline(Table.index.get_loc(train_end) - 4)

	plt.legend()
	plt.title(m)
	plt.show()
	
	# save all the plots
	for m in range(estimation.shape[1]):
		plt.close()
		if flag_real_plot:
			plt.plot(Table[test_start:].iloc[4:,m].values,label='real', c='red')
		plt.plot(estimation[4:,m], label='prediction', c='b')

		under_line = estimation[4:,m] - sigma[4:,m]
		over_line = estimation[4:,m] + sigma[4:,m]
		plt.fill_between(range(len(estimation[4:,m])),under_line, over_line, color='b', alpha=0.2) #std curves.

		if train_end in Table.index :
			plt.axvline(Table.index.get_loc(train_end) - 4)

		plt.legend()
		plt.title(m)
		plt.savefig(pt+'/'+str(m)+'.png')
	

class TransformerPredictionPlugin:
    def input(self, filename):
        self.parameters = dict()
        infile = open(filename, 'r')
        for line in infile:
           contents = line.strip().split('\t')
           self.parameters[contents[0]] = contents[1]

        self.data_path = PyPluMA.prefix()+"/"+self.parameters["datapath"]
        self.data_covariates_path = None
        self.date_column = None
        self.model_config_path = PyPluMA.prefix()+"/"+self.parameters["modelconfig"]
        self.meta_information_path = PyPluMA.prefix()+"/"+self.parameters["metainformation"]
        self.prediction_name = self.parameters["predictionname"]
        self.flag_real_plot = False
        self.steps = int(self.parameters["steps"])

    def run(self):
        pass

    def output(self, filename):
        main(self.data_path, self.data_covariates_path, self.date_column, filename, self.prediction_name, self.model_config_path, self.meta_information_path, self.steps, self.flag_real_plot)
