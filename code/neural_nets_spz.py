import torch 
from torch.utils.data import Dataset, DataLoader


class NeuralNet(torch.nn.Module):

    def __init__(self, input_size, hidden_size=32, output_size=1):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.output_size = output_size
        self.lin1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.act1 = torch.nn.SELU()
        #self.do1 = torch.nn.Dropout(0.2)
        self.lin2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.act2 = torch.nn.SELU()
        #self.do2 = torch.nn.Dropout(0.2)
        self.lin3 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.act3 = torch.nn.SELU()
        #self.do3 = torch.nn.Dropout(0.2)
        self.linfinal = torch.nn.Linear(self.hidden_size, output_size)

        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.zeros_(self.lin1.bias)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        torch.nn.init.zeros_(self.lin2.bias)
        torch.nn.init.xavier_uniform_(self.lin3.weight)
        torch.nn.init.zeros_(self.lin3.bias)
        torch.nn.init.xavier_uniform_(self.linfinal.weight)
        torch.nn.init.zeros_(self.linfinal.bias)
        self.double()

    def forward(self, x):
        x = self.lin1(x)
        x = self.act1(x)
        #x = self.do1(x)
        x = self.lin2(x)
        x = self.act2(x)
        #x = self.do2(x)
        x = self.lin3(x)
        x = self.act3(x)
        #x = self.do3(x)
        output = self.linfinal(x)
        return output


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)
    print('worker seed', worker_seed)


class RedshiftEstimatorANN(RedshiftEstimator):
    
    def __init__(self, *args, feature_keys=None, rng=None, learning_rate=0.005, batch_size=512,
                 **kwargs):
        self.rng = rng
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.feature_keys = feature_keys
        super().__init__(*args, **kwargs)
        if self.train_mode:
            assert rng is not None, "Must pass RNG for ANN!"
        if self.train_mode:
            self.set_up_data()

    
    def set_up_data(self, frac_valid=0.2):
        # switched to passing in validation data
        # N_train = self.X_train.shape[0]
        # # assign unique ints to the training set
        # random_ints = self.rng.choice(range(N_train), size=N_train, replace=False)
        # # split into actual training and validation subset
        # int_divider = int(frac_valid*N_train)
        # idx_valid = np.where(random_ints < int_divider)[0]
        # idx_train = np.where(random_ints >= int_divider)[0]
        # print("N_train:", len(idx_train), "N_valid:", len(idx_valid))

        # self.X_train_sub = self.X_train[idx_train]
        # self.Y_train_sub = self.Y_train[idx_train]
        # self.X_valid = self.X_train[idx_valid]
        # self.Y_valid = self.Y_train[idx_valid]

        # TODO just did this for now because abandoning train_sub 
        # now that have separate valid, but annoying to redo so keeping in case
        self.X_train_sub = self.X_train
        self.Y_train_sub = self.Y_train

        self.scale_x()
        self.scale_y()

        self.dataset_train = DataSet(self.X_train_sub_scaled, self.Y_train_sub_scaled)
        self.data_loader_train = DataLoader(self.dataset_train, 
                                    batch_size=self.batch_size, shuffle=True,
                                    worker_init_fn=seed_worker,
                                    num_workers=0)

        self.dataset_valid = DataSet(self.X_valid_scaled, self.Y_valid_scaled)
        self.data_loader_valid = DataLoader(self.dataset_valid, 
                                    batch_size=self.batch_size, shuffle=True,
                                    worker_init_fn=seed_worker,
                                    num_workers=0)


    def scale_x(self):
        N_feat = self.X_train.shape[1]
        # for all columns with word redshift, do not scale
        if self.feature_keys is not None:
            idx_features_to_scale = []
            for idx, feat_key in enumerate(self.feature_keys):
                if 'redshift' not in feat_key:
                    idx_features_to_scale.append(idx)
        else:
            idx_features_to_scale = np.arange(N_feat)
        print("Features to scale:", idx_features_to_scale)
        print("Feature keys:", self.feature_keys)
        self.scaler_x = ColumnTransformer([("standard", StandardScaler(), 
                                           np.array(idx_features_to_scale))], remainder='passthrough')
        #self.scaler_x = StandardScaler()
        self.scaler_x.fit(self.X_train_sub)
        self.X_train_sub_scaled = self.scaler_x.transform(self.X_train_sub)
        self.X_valid_scaled = self.scaler_x.transform(self.X_valid)
        # print(self.X_train_sub[0])
        # print(self.X_train_sub_scaled[0])


    def scale_y(self):
        self.scaler_y = StandardScaler(with_mean=False, with_std=False)
        self.scaler_y.fit(np.atleast_2d(self.Y_train_sub).T)
        self.Y_train_sub_scaled = self.scaler_y.transform(np.atleast_2d(self.Y_train_sub).T)
        self.Y_valid_scaled = self.scaler_y.transform(np.atleast_2d(self.Y_valid).T)
        # print(self.Y_train_sub[:5])
        # print(self.Y_train_sub_scaled[:5])


    def apply(self):
        print("Applying")
        self.X_apply_scaled = self.scaler_x.transform(self.X_apply)
        self.Y_hat_apply, self.sigma_z = self.predict(self.X_apply_scaled)
        return self.Y_hat_apply, self.sigma_z


    def train_one_epoch(self, epoch_index):
        running_loss_train = 0.
        running_loss_valid = 0.
        losses_train = []
        for i, data in enumerate(self.data_loader_train):
            x, y = data

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()
            # Make predictions for this batch
            y_pred = self.model(x.double())
            # Compute the loss and its gradients
            # squeeze all in case they are 1-dim
            loss = self.criterion(y_pred.squeeze(), y.squeeze())
            loss.backward()
            # Adjust learning weights
            self.optimizer.step()
            # Gather data and report
            running_loss_train += loss.item()
            losses_train.append(loss.item())

        self.model.eval()
        for i, data_val in enumerate(self.data_loader_valid):
            x, y = data_val
            y_pred = self.model(x.double())
            loss = self.criterion(y_pred.squeeze(), y.squeeze())
            running_loss_valid += loss.item()

        #print(np.mean(losses_train), np.min(losses_train), np.max(losses_train))
        last_loss_train = running_loss_train / len(self.data_loader_train)
        last_loss_valid = running_loss_valid / len(self.data_loader_valid)
        print(f"Training epoch {epoch_index}, training loss {last_loss_train:.3f}, validation loss {last_loss_valid:.3f}")
        return last_loss_train, last_loss_valid



    def train(self, hidden_size=512, max_epochs=30, 
              fn_model=None, save_at_min_loss=True):

        input_size = self.X_train.shape[1] # number of features
        output_size = 1 # 1 redshift estimate
        self.model = NeuralNet(input_size, hidden_size=hidden_size, output_size=output_size)

        self.criterion = torch.nn.MSELoss()
        #self.criterion = torch.nn.GaussianNLLLoss()
        def loss_dz(output, target):
            loss = torch.mean(torch.divide(torch.abs(output - target), torch.add(target, 1.0)))
            return loss
        #self.criterion = loss_dz
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        self.loss_train = []
        self.loss_valid = []
        self.model.train()
        loss_valid_min = np.inf
        epoch_best = None
        state_dict_best = None
        for epoch_index in range(max_epochs):
            last_loss_train, last_loss_valid = self.train_one_epoch(epoch_index)
            #print(last_loss, loss_min)
            if save_at_min_loss and last_loss_valid < loss_valid_min:
                #print(last_loss, loss_min)
                state_dict_best = self.model.state_dict()
                #print(state_dict_best)
                epoch_best = epoch_index
                loss_valid_min = last_loss_valid
            self.loss_train.append(last_loss_train)
            self.loss_valid.append(last_loss_valid)
        
        print('Epoch best:', epoch_best)
        # revert to state dict for model with lowest loss
        if save_at_min_loss:
            self.model.load_state_dict(state_dict_best)
        # if fn_model is not None:
        #     # if save_at_min_loss=False, will just save the last epoch 
        #     self.save_model(fn_model, epoch=epoch_best)


    def predict(self, X_input_scaled):
        self.model.eval()
        with torch.no_grad():
            y_pred_scaled = self.model(torch.from_numpy(X_input_scaled).double())

        y_pred_scaled = y_pred_scaled.squeeze().numpy()
        print(y_pred_scaled.shape)
        y_pred = np.squeeze(self.scaler_y.inverse_transform(np.atleast_2d(y_pred_scaled).T))
        #y_pred = y_pred_scaled
        print(y_pred.shape)
        sigma = [np.NaN]*len(y_pred) 
        return y_pred, sigma


    def save_model(self, fn_model, epoch=None):
        if epoch is None:
            epoch = len(self.loss_valid)
        save_dict = {
                    'input_size': self.model.input_size,
                    'hidden_size': self.model.hidden_size,
                    'output_size': self.model.output_size,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scaler_x': self.scaler_x,
                    'scaler_y': self.scaler_y,
                    'loss_train': self.loss_train,
                    'loss_valid': self.loss_valid,
                    'epoch': epoch
                    }
        torch.save(save_dict, fn_model)


    def load_model(self, fn_model):
        model_checkpoint = torch.load(fn_model)
        if 'output_size' in model_checkpoint:
            output = model_checkpoint['output_size']
        else:
            # for back-compatibility
            output = 1
        self.model = NeuralNet(model_checkpoint['input_size'], hidden_size=model_checkpoint['hidden_size'],
                               output_size=output)
        self.model.load_state_dict(model_checkpoint['model_state_dict'])
        self.model.eval()
        self.scaler_x = model_checkpoint['scaler_x']
        self.scaler_y = model_checkpoint['scaler_y']
        if 'loss_train' in model_checkpoint:
            self.loss_train = model_checkpoint['loss_train']
        if 'loss_valid' in model_checkpoint:
            self.loss_valid = model_checkpoint['loss_valid']
        if 'loss' in model_checkpoint:
            self.loss = model_checkpoint['loss']        
        self.epoch = model_checkpoint['epoch']



class RedshiftEstimatorANN2class(RedshiftEstimator):
    
    def __init__(self, *args, rng=None, learning_rate=0.005, batch_size=512, **kwargs):
        self.rng = rng
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        super().__init__(*args, **kwargs)
        if self.train_mode:
            assert rng is not None, "Must pass RNG for ANN!"
        if self.train_mode:
            self.set_up_data()

    
    def set_up_data(self):

        # TODO just did this for now because abandoning train_sub 
        # now that have separate valid, but annoying to redo so keeping in case
        self.X_train_sub = self.X_train
        self.Y_train_sub = self.Y_train

        self.scale_x()
        print(self.Y_train_sub)

        self.dataset_train = DataSet(self.X_train_sub_scaled, self.Y_train_sub)
        self.data_loader_train = DataLoader(self.dataset_train, 
                                    batch_size=self.batch_size, shuffle=True,
                                    worker_init_fn=seed_worker,
                                    num_workers=0)

        self.dataset_valid = DataSet(self.X_valid_scaled, self.Y_valid)
        self.data_loader_valid = DataLoader(self.dataset_valid, 
                                    batch_size=self.batch_size, shuffle=True,
                                    worker_init_fn=seed_worker,
                                    num_workers=0)


    def scale_x(self):
        N_feat = self.X_train.shape[1]
        # assumes redshift_qsoc is first column
        self.scaler_x = ColumnTransformer([("standard", StandardScaler(), np.arange(1,N_feat))], remainder='passthrough')
        #self.scaler_x = StandardScaler()
        self.scaler_x.fit(self.X_train_sub)
        self.X_train_sub_scaled = self.scaler_x.transform(self.X_train_sub)
        self.X_valid_scaled = self.scaler_x.transform(self.X_valid)


    def apply(self):
        print("Applying")
        self.X_apply_scaled = self.scaler_x.transform(self.X_apply)
        self.Y_hat_apply, self.sigma_z = self.predict(self.X_apply_scaled)
        return self.Y_hat_apply, self.sigma_z


    def train_one_epoch(self, epoch_index):
        running_loss_train = 0.
        running_loss_valid = 0.
        losses_train = []
        for i, data in enumerate(self.data_loader_train):
            x, y = data

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()
            # Make predictions for this batch
            y_pred = self.model(x.double())
            # Compute the loss and its gradients
            # squeeze all in case they are 1-dim
            loss = self.criterion(y_pred.squeeze(), y.squeeze())
            loss.backward()
            # Adjust learning weights
            self.optimizer.step()
            # Gather data and report
            running_loss_train += loss.item()
            losses_train.append(loss.item())

        self.model.eval()
        for i, data_val in enumerate(self.data_loader_valid):
            x, y = data_val
            y_pred = self.model(x.double())
            loss = self.criterion(y_pred.squeeze(), y.squeeze())
            running_loss_valid += loss.item()

        #print(np.mean(losses_train), np.min(losses_train), np.max(losses_train))
        last_loss_train = running_loss_train / len(self.data_loader_train)
        last_loss_valid = running_loss_valid / len(self.data_loader_valid)
        print(f"Training epoch {epoch_index}, training loss {last_loss_train:.3f}, validation loss {last_loss_valid:.3f}")
        return last_loss_train, last_loss_valid



    def train(self, hidden_size=512, max_epochs=20, 
              fn_model=None, save_at_min_loss=True):

        input_size = self.X_train.shape[1] # number of features
        output_size = 1 # 1 redshift estimate
        self.model = NeuralNet(input_size, hidden_size=hidden_size, output_size=output_size)

        # binary cross entropy including a sigmoid to squeeze output of NN into 0-1
        # pos_weight: https://discuss.pytorch.org/t/bcewithlogitsloss-calculating-pos-weight/146336/3
        N_pos = np.sum(self.Y_train)
        N = len(self.Y_train)
        pos_weight = N/N_pos - 1 #this is equiv to N_neg/N_pos, as that forum says
        
        print('pos_weight:', N_pos, N, pos_weight)
        #pos_weight *= 1.3
        #print("INFLATING POS_WEIGHT, now", pos_weight)

        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        self.loss_train = []
        self.loss_valid = []
        self.model.train()
        loss_valid_min = np.inf
        epoch_best = None
        state_dict_best = None
        for epoch_index in range(max_epochs):
            last_loss_train, last_loss_valid = self.train_one_epoch(epoch_index)
            #print(last_loss, loss_min)
            if save_at_min_loss and last_loss_valid < loss_valid_min:
                #print(last_loss, loss_min)
                state_dict_best = self.model.state_dict()
                #print(state_dict_best)
                epoch_best = epoch_index
                loss_valid_min = last_loss_valid
            self.loss_train.append(last_loss_train)
            self.loss_valid.append(last_loss_valid)
        
        print('Epoch best:', epoch_best)
        # revert to state dict for model with lowest loss
        if save_at_min_loss:
            self.model.load_state_dict(state_dict_best)
        # if fn_model is not None:
        #     # if save_at_min_loss=False, will just save the last epoch 
        #     self.save_model(fn_model, epoch=epoch_best)


    def predict(self, X_input_scaled):
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(torch.from_numpy(X_input_scaled).double())

        # to get from logits to probabilities
        # https://sebastianraschka.com/blog/2022/losses-learned-part1.html
        c_pred = torch.sigmoid(y_pred)
        c_pred = c_pred.squeeze().numpy()
        print(c_pred.shape)
        sigma = [np.NaN]*len(y_pred) 
        return c_pred, sigma


    def save_model(self, fn_model, epoch=None):
        if epoch is None:
            epoch = len(self.loss_valid)
        save_dict = {
                    'input_size': self.model.input_size,
                    'hidden_size': self.model.hidden_size,
                    'output_size': self.model.output_size,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scaler_x': self.scaler_x,
                    'loss_train': self.loss_train,
                    'loss_valid': self.loss_valid,
                    'epoch': epoch
                    }
        torch.save(save_dict, fn_model)


    def load_model(self, fn_model):
        model_checkpoint = torch.load(fn_model)
        if 'output_size' in model_checkpoint:
            output = model_checkpoint['output_size']
        else:
            # for back-compatibility
            output = 1
        self.model = NeuralNet(model_checkpoint['input_size'], hidden_size=model_checkpoint['hidden_size'],
                               output_size=output)
        self.model.load_state_dict(model_checkpoint['model_state_dict'])
        self.model.eval()
        self.scaler_x = model_checkpoint['scaler_x']
        if 'loss_train' in model_checkpoint:
            self.loss_train = model_checkpoint['loss_train']
        if 'loss_valid' in model_checkpoint:
            self.loss_valid = model_checkpoint['loss_valid']
        if 'loss' in model_checkpoint:
            self.loss = model_checkpoint['loss']        
        self.epoch = model_checkpoint['epoch']


class RedshiftEstimatorANNmulticlass(RedshiftEstimator):
    
    def __init__(self, *args, rng=None, learning_rate=0.005, batch_size=512, 
                 N_classes=1, **kwargs):
        self.rng = rng
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.N_classes = N_classes
        super().__init__(*args, **kwargs)
        if self.train_mode:
            assert rng is not None, "Must pass RNG for ANN!"
        if self.train_mode:
            self.set_up_data()

    
    def set_up_data(self):

        # TODO just did this for now because abandoning train_sub 
        # now that have separate valid, but annoying to redo so keeping in case
        self.X_train_sub = self.X_train
        self.Y_train_sub = self.Y_train

        self.scale_x()
        print(self.Y_train_sub)

        self.dataset_train = DataSet(self.X_train_sub_scaled, self.Y_train_sub)
        self.data_loader_train = DataLoader(self.dataset_train, 
                                    batch_size=self.batch_size, shuffle=True,
                                    worker_init_fn=seed_worker,
                                    num_workers=0)

        self.dataset_valid = DataSet(self.X_valid_scaled, self.Y_valid)
        self.data_loader_valid = DataLoader(self.dataset_valid, 
                                    batch_size=self.batch_size, shuffle=True,
                                    worker_init_fn=seed_worker,
                                    num_workers=0)


    def scale_x(self):
        N_feat = self.X_train.shape[1]
        # assumes redshift_qsoc is first column
        self.scaler_x = ColumnTransformer([("standard", StandardScaler(), np.arange(1,N_feat))], remainder='passthrough')
        #self.scaler_x = StandardScaler()
        self.scaler_x.fit(self.X_train_sub)
        self.X_train_sub_scaled = self.scaler_x.transform(self.X_train_sub)
        self.X_valid_scaled = self.scaler_x.transform(self.X_valid)


    def apply(self):
        print("Applying")
        self.X_apply_scaled = self.scaler_x.transform(self.X_apply)
        self.Y_hat_apply_raw, self.sigma_z = self.predict(self.X_apply_scaled)
        ### _,pred = torch.max(out, dim=1)
        # the raw data is the probabilities; argmax to get highest-prob zbin
        self.Y_hat_apply = self.Y_hat_apply_raw.argmax(axis=1)
        print(self.Y_hat_apply_raw[0])
        print(self.Y_hat_apply[0])
        return self.Y_hat_apply, self.sigma_z


    def train_one_epoch(self, epoch_index):
        running_loss_train = 0.
        running_loss_valid = 0.
        losses_train = []
        for i, data in enumerate(self.data_loader_train):
            x, y = data

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()
            # Make predictions for this batch
            y_pred = self.model(x.double())
            # Compute the loss and its gradients
            # squeeze all in case they are 1-dim
            loss = self.criterion(y_pred.squeeze(), y.squeeze())
            loss.backward()
            # Adjust learning weights
            self.optimizer.step()
            # Gather data and report
            running_loss_train += loss.item()
            losses_train.append(loss.item())

        self.model.eval()
        for i, data_val in enumerate(self.data_loader_valid):
            x, y = data_val
            y_pred = self.model(x.double())
            loss = self.criterion(y_pred.squeeze(), y.squeeze())
            running_loss_valid += loss.item()

        #print(np.mean(losses_train), np.min(losses_train), np.max(losses_train))
        last_loss_train = running_loss_train / len(self.data_loader_train)
        last_loss_valid = running_loss_valid / len(self.data_loader_valid)
        print(f"Training epoch {epoch_index}, training loss {last_loss_train:.3f}, validation loss {last_loss_valid:.3f}")
        return last_loss_train, last_loss_valid


    def train(self, hidden_size=512, max_epochs=20, 
              fn_model=None, save_at_min_loss=True):

        input_size = self.X_train.shape[1] # number of features
        output_size = self.N_classes # 1 redshift estimate
        self.model = NeuralNet(input_size, hidden_size=hidden_size, output_size=output_size)

        # binary cross entropy including a sigmoid to squeeze output of NN into 0-1
        # pos_weight: https://discuss.pytorch.org/t/bcewithlogitsloss-calculating-pos-weight/146336/3
        N = len(self.Y_train)
        weights = [N/np.sum(self.Y_train==i)-1 for i in range(self.N_classes)]
        print('frac in each class:', [np.sum(self.Y_train==i)/N for i in range(self.N_classes)])
        print('weights', weights)
        self.criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        self.loss_train = []
        self.loss_valid = []
        self.model.train()
        loss_valid_min = np.inf
        epoch_best = None
        state_dict_best = None
        for epoch_index in range(max_epochs):
            last_loss_train, last_loss_valid = self.train_one_epoch(epoch_index)
            #print(last_loss, loss_min)
            if save_at_min_loss and last_loss_valid < loss_valid_min:
                #print(last_loss, loss_min)
                state_dict_best = self.model.state_dict()
                #print(state_dict_best)
                epoch_best = epoch_index
                loss_valid_min = last_loss_valid
            self.loss_train.append(last_loss_train)
            self.loss_valid.append(last_loss_valid)
        
        print('Epoch best:', epoch_best)
        # revert to state dict for model with lowest loss
        if save_at_min_loss:
            self.model.load_state_dict(state_dict_best)
        # if fn_model is not None:
        #     # if save_at_min_loss=False, will just save the last epoch 
        #     self.save_model(fn_model, epoch=epoch_best)


    def predict(self, X_input_scaled):
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(torch.from_numpy(X_input_scaled).double())

        # to get from logits to probabilities
        # https://sebastianraschka.com/blog/2022/losses-learned-part1.html
        c_pred = torch.sigmoid(y_pred)
        c_pred = c_pred.squeeze().numpy()
        print(c_pred.shape)
        sigma = [np.NaN]*len(y_pred) 
        return c_pred, sigma


    def save_model(self, fn_model, epoch=None):
        if epoch is None:
            epoch = len(self.loss_valid)
        save_dict = {
                    'input_size': self.model.input_size,
                    'hidden_size': self.model.hidden_size,
                    'output_size': self.model.output_size,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scaler_x': self.scaler_x,
                    'loss_train': self.loss_train,
                    'loss_valid': self.loss_valid,
                    'epoch': epoch
                    }
        torch.save(save_dict, fn_model)


    def load_model(self, fn_model):
        model_checkpoint = torch.load(fn_model)
        if 'output_size' in model_checkpoint:
            output = model_checkpoint['output_size']
        else:
            # for back-compatibility
            output = 1
        self.model = NeuralNet(model_checkpoint['input_size'], hidden_size=model_checkpoint['hidden_size'],
                               output_size=output)
        self.model.load_state_dict(model_checkpoint['model_state_dict'])
        self.model.eval()
        self.scaler_x = model_checkpoint['scaler_x']
        if 'loss_train' in model_checkpoint:
            self.loss_train = model_checkpoint['loss_train']
        if 'loss_valid' in model_checkpoint:
            self.loss_valid = model_checkpoint['loss_valid']
        if 'loss' in model_checkpoint:
            self.loss = model_checkpoint['loss']        
        self.epoch = model_checkpoint['epoch']


class DataSet(Dataset):

    def __init__(self, X, Y, y_var=None, randomize=True):
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.y_var = y_var
        if len(self.X) != len(self.Y):
            raise Exception("The length of X does not match the length of Y")
        if y_var is not None:
            self.y_var = np.array(self.y_var)
            if len(self.X) != len(self.y_var):
                raise Exception("The length of X does not match the length of y_var")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        _x = self.X[index]
        _y = self.Y[index]
        if self.y_var is not None:
            _y_var = self.y_var[index]
            return _x, _y, _y_var
        return _x, _y