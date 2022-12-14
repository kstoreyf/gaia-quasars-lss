import numpy as np
from numpy.random import default_rng

import astropy
import astropy.units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.table import Table, join

from dustmaps.sfd import SFDQuery
from sklearn.neighbors import KDTree

import torch 
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.compose import ColumnTransformer

import utils


def main():

    G_max = 20.5
    rng = default_rng()
    redshift_estimator_name = 'kNN'

    redshift_estimator_dict = {'kNN': RedshiftEstimatorkNN,
                               'ANN': RedshiftEstimatorANN
                               }
    redshift_estimator_kwargs_dict = {'kNN': {'K': 11},
                                      'ANN': {'rng': rng}
                                      }
    redshift_estimator_class = redshift_estimator_dict[redshift_estimator_name]                        
    redshift_estimator_kwargs = redshift_estimator_kwargs_dict[redshift_estimator_name]

    # save name
    #save_tag = '_lr0.005'
    save_tag = '_scaledNOqsoc'
    fn_spz = f'../data/redshifts_spz_{redshift_estimator_name}_G{G_max}{save_tag}.fits'
    overwrite = True

    # Load data
    print("Loading data")
    fn_gaia = '../data/gaia_clean.fits'
    tab_gaia = utils.load_table(fn_gaia)
    # TEST ONLY W SMALL AMOUNT
    #tab_gaia = tab_gaia[np.random.randint(0, len(tab_gaia), size=10000)]
    N_gaia = len(tab_gaia)
    print(f"N of clean gaia catalog: {N_gaia}")

    # Make Gmax cut, because will be more robust that way if we cut first
    i_makeGcut = tab_gaia['phot_g_mean_mag'] < G_max
    tab_gaia = tab_gaia[i_makeGcut]
    print(f"N after G_max cut of G<{G_max}:", len(tab_gaia))

    # Construct full feature matrix
    print("Constructing feature matrix")
    feature_keys = ['redshift_qsoc', 'ebv', 'g_rp', 'bp_g', 'bp_rp', 'g_w1', 'w1_w2', 'phot_g_mean_mag']
    X_gaia, idx_goodfeat = construct_X(tab_gaia, feature_keys)
    # these indexes are the ones in our final sample
    X_gaia = X_gaia[idx_goodfeat]
    tab_gaia = tab_gaia[idx_goodfeat]
    print("N after throwing out bad features:", len(tab_gaia))
    i_has_sdss_redshift = np.isfinite(tab_gaia['sdss_Z'])
    print("N with SDSS redshifts:", np.sum(i_has_sdss_redshift))

    # Split training (where have SDSS redshifts) and not
    X_train = X_gaia[i_has_sdss_redshift]
    # Apply to all, including those with SDSS redshifts (for consistency)
    X_apply = X_gaia
    Y_train = tab_gaia[i_has_sdss_redshift]['sdss_Z']
    print(f"X_train: {X_train.shape}, Y_train: {Y_train.shape}, X_apply: {X_apply.shape}")
    assert X_train.shape[0]==Y_train.shape[0], "X and Y must have same length!"

    # Run redshift estimation
    print("Running redshift estimation")
    # cross_validate(redshift_estimator_class, 
    #                X_train, Y_train,
    #                redshift_estimator_kwargs, rng)
    redshift_estimator = redshift_estimator_class(X_train, Y_train, X_apply, **redshift_estimator_kwargs)
    redshift_estimator.train()
    redshift_estimator.apply()

    # Save results
    print("Save results")
    columns_to_keep = ['source_id', 'sdss_OBJID', 'phot_g_mean_mag', 'redshift_qsoc', 'sdss_Z']
    tab_gaia.keep_columns(columns_to_keep)
    tab_gaia['redshift_spz'] = redshift_estimator.Y_hat_apply
    tab_gaia['redshift_spz_err'] = redshift_estimator.sigma_z
    tab_gaia.write(fn_spz, overwrite=overwrite)
    print(f"Wrote specphotozs to {fn_spz}!")


def construct_X(tab, feature_keys):

    X = []
    for feature_key in feature_keys:
        X.append(tab[feature_key])
    X = np.array(X).T
    idx_goodfeat = np.all(np.isfinite(X), axis=1)
    
    return X, idx_goodfeat



def cross_validate(redshift_estimator_class,
                   X_train, Y_train,
                   redshift_estimator_kwargs, rng, n_samples=8):

    z_errs_close = [0.1, 0.2]

    print("Cross validating")
    i_sample_vals = np.arange(n_samples)
    # high is exclusive
    i_samples_loo = rng.integers(low=0, high=n_samples, size=X_train.shape[0])

    Y_hat = np.empty(X_train.shape[0])
    sigma_z = np.empty(X_train.shape[0])

    print("ONE CROSS VAL FOR NOW")
    #for i_sample_val in i_sample_vals:
    for i_sample_val in [i_sample_vals[0]]:
        print(f"Leave-one-out sample {i_sample_val}")
        idx_train = i_samples_loo != i_sample_val
        idx_test = i_samples_loo == i_sample_val
        X_train_loo, Y_train_loo = X_train[idx_train], Y_train[idx_train]
        X_test_loo, Y_test_loo = X_train[idx_test], Y_train[idx_test]

        redshift_estimator = redshift_estimator_class(X_train_loo, Y_train_loo, X_test_loo,
                                                      **redshift_estimator_kwargs)

        redshift_estimator.train()
        Y_hat_test_loo, sigma_z_test_loo = redshift_estimator.apply()

        #tree_loo = self.build_tree(X_train_loo)
        # TODO: only passing X train to check things, can delete that kwarg
        #Y_hat_valid_loo, sigma_z_valid_loo = self.get_median_kNNs(X_valid_loo, Y_train=Y_train_loo)
        
        Y_hat[idx_test] = Y_hat_test_loo
        sigma_z[idx_test] = sigma_z_test_loo

        for z_err_close in z_errs_close:
            frac_recovered = get_fraction_recovered(Y_test_loo, Y_hat_test_loo, z_err_close)
            print(rf"Fraction recovered with $\delta z$<{z_err_close}: {frac_recovered:.3f}")

    for z_err_close in z_errs_close:
        frac_recovered = get_fraction_recovered(Y_train, Y_hat, z_err_close)
        print(rf"Overall fraction recovered with $\delta z$<{z_err_close}: {frac_recovered:.3f}")


def get_fraction_recovered(Y_true, Y_hat, z_err_close):
    return np.sum(np.abs(Y_true - Y_hat) < z_err_close) / len(Y_true)



class RedshiftEstimator():

    def __init__(self, X_train, Y_train,
                       X_apply):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_apply = X_apply


    def train(self):
        pass 


    def predict(self):
        pass


    def apply(self):
        pass




class RedshiftEstimatorkNN(RedshiftEstimator):

    
    def __init__(self, *args, K=11, **kwargs):
        super().__init__(*args, **kwargs)
        self.K = K
        self.scale_x()


    def scale_x(self):
        # mean and stdev
        #self.scaler_x = StandardScaler() # TODO revisit !! 
        #does nothing while keeping notation consistent
        #self.scaler_x = StandardScaler(with_mean=False, with_std=False) 
        # scales all besides 1st column, QSOC
        N_feat = self.X_train.shape[1]
        self.scaler_x = ColumnTransformer([("standard", StandardScaler(), np.arange(1,N_feat))], remainder='passthrough')
        self.scaler_x.fit(self.X_train)
        self.X_train_scaled = self.scaler_x.transform(self.X_train)
        self.X_apply_scaled = self.scaler_x.transform(self.X_apply)


    def train(self):
        print("Training")
        self.tree_train = self.build_tree(self.X_train_scaled)


    def apply(self):
        print("Applying")
        self.Y_hat_apply, self.sigma_z = self.predict(self.X_apply_scaled)
        return self.Y_hat_apply, self.sigma_z


    def build_tree(self, X):
        print("Building kdTree")
        return KDTree(X)


    def predict(self, X_input_scaled):
        print("Getting median Z of nearest neighbors")
        dists, inds = self.tree_train.query(X_input_scaled, k=self.K+1)
        # if nearest neighbor is itself (dist~0), exclude that one;
        # to do this, need to get more neighbors than maybe necessary
        # to keep it at K overall
        dist_min = 1e-8 #hack
        idx_nearest_dist0 = dists[:,0] < dist_min
        print(f"A fraction {np.sum(idx_nearest_dist0)/len(idx_nearest_dist0):.3f} of objects have nearest neighbor w dist zero; cutting these from median")
        inds_nodist0 = np.empty((inds.shape[0], self.K), dtype=int)
        inds_nodist0[idx_nearest_dist0] = inds[idx_nearest_dist0,1:]
        inds_nodist0[~idx_nearest_dist0] = inds[~idx_nearest_dist0,:-1]
        low_z, Y_hat, up_z = np.percentile(self.Y_train[inds_nodist0], (2.5, 50, 97.5), axis=1)
        sigma_z = (up_z - low_z)/4
        return Y_hat, sigma_z



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
    
    def __init__(self, *args, rng=None, batch_size=512, **kwargs):
        self.rng = rng
        self.batch_size = batch_size
        assert rng is not None, "Must pass RNG for ANN!"
        super().__init__(*args, **kwargs)
        self.set_up_data()

    
    def set_up_data(self, frac_valid=0.2):
        N_train = self.X_train.shape[0]
        # assign unique ints to the training set
        random_ints = self.rng.choice(range(N_train), size=N_train, replace=False)
        # split into actual training and validation subset
        int_divider = int(frac_valid*N_train)
        idx_valid = np.where(random_ints < int_divider)[0]
        idx_train = np.where(random_ints >= int_divider)[0]
        print("N_train:", len(idx_train), "N_valid:", len(idx_valid))

        self.X_train_sub = self.X_train[idx_train]
        self.Y_train_sub = self.Y_train[idx_train]
        self.X_valid = self.X_train[idx_valid]
        self.Y_valid = self.Y_train[idx_valid]

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
        # assumes redshift_qsoc is first column
        self.scaler_x = ColumnTransformer([("standard", StandardScaler(), np.arange(1,N_feat))], remainder='passthrough')
        #self.scaler_x = StandardScaler()
        self.scaler_x.fit(self.X_train_sub)
        self.X_train_sub_scaled = self.scaler_x.transform(self.X_train_sub)
        self.X_valid_scaled = self.scaler_x.transform(self.X_valid)
        self.X_apply_scaled = self.scaler_x.transform(self.X_apply)
        print(self.X_train_sub[0])
        print(self.X_train_sub_scaled[0])


    def scale_y(self):
        
        
        self.scaler_y = StandardScaler(with_mean=False, with_std=False)
        self.scaler_y.fit(np.atleast_2d(self.Y_train_sub).T)
        self.Y_train_sub_scaled = self.scaler_y.transform(np.atleast_2d(self.Y_train_sub).T)
        self.Y_valid_scaled = self.scaler_y.transform(np.atleast_2d(self.Y_valid).T)
        print(self.Y_train_sub[:5])
        print(self.Y_train_sub_scaled[:5])


    def apply(self):
        print("Applying")
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



    def train(self, hidden_size=512, max_epochs=100, learning_rate=0.005, 
              fn_model=None, save_at_min_loss=True):

        input_size = self.X_train.shape[1] # number of features
        output_size = 1 # 1 redshift estimate
        self.model = NeuralNet(input_size, hidden_size=hidden_size, output_size=output_size)

        self.criterion = torch.nn.MSELoss()
        #self.criterion = torch.nn.GaussianNLLLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

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




if __name__=='__main__':
    main()