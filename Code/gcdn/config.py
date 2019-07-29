class Config(object):
    
    def __init__(self):

        # directories
        self.save_dir = ''       
        self.log_dir = ''
        self.train_data_file = ''
        self.val_data_file = ''

        # input
        self.patch_size = [42, 42, 1]
        self.N          = self.patch_size[0]*self.patch_size[1]
        # no. of layers
        self.pre_n_layers       = 3
        self.pregconv_n_layers  = 1
        self.hpf_n_layers       = 3
        self.lpf_n_layers       = 3
        self.prox_n_layers      = 4
        # no. of features
        self.Nfeat              = 132 # must be multiple of 3
        self.pre_Nfeat          = self.Nfeat/3
        self.pre_fnet_Nfeat     = self.pre_Nfeat
        self.prox_fnet_Nfeat    = self.Nfeat
        self.hpf_fnet_Nfeat     = self.Nfeat
        # gconv params
        self.rank_theta         = 11    
        self.stride             = self.Nfeat/3
        self.stride_pregconv    = self.Nfeat/3
        self.min_nn             = 16 +8

        # learning
        self.batch_size = 12
        self.grad_accum = 1
        self.N_iter = 400000
        self.starter_learning_rate = 1e-4
        self.end_learning_rate = 1e-5
        self.decay_step = 1000
        self.decay_rate = (self.end_learning_rate / self.starter_learning_rate)**(float(self.decay_step) / self.N_iter)
        self.Ngpus = 2

        # debugging
        self.save_every_iter = 250
        self.summaries_every_iter = 5
        self.validate_every_iter = 100
        self.test_every_iter = 250

        # testing
        self.minisize = 49*3 # must be integer multiple of search window
        self.search_window = [49,49]
        self.searchN = self.search_window[0]*self.search_window[1]

        # noise std
        self.sigma = 25
