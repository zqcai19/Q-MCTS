class Arguments:
    def __init__(self):
        self.n_qubits   = 7
        self.a_insize   = 74
        self.v_insize   = 35
        self.t_insize   = 300
        self.a_hidsize  = 6 #8
        self.v_hidsize  = 3 #4
        self.t_hidsize  = 12 #128
        # self.a_projsize = 6
        # self.v_projsize = 3
        # self.t_projsize = 12
        self.device     = 'cpu'
        self.clr        = 0.005
        self.qlr        = 0.05
        self.epochs     = 5
        self.batch_size = 32
        self.test_batch_size = 673
