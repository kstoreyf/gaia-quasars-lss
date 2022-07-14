def get_fraction_recovered(Y_true, Y_hat, z_err_close):
        return np.sum(np.abs(Y_true - Y_hat) < z_err_close) / len(Y_true)
