import pickle
data = pickle.load(open('../data/TSMC2014_NYC' + '.pk', 'rb'),encoding='bytes')

    

class Setting():
    def __init__(self):
        self.loc_emb_size = 424
        self.tim_emb_size = 0
        self.hidden_size = 424
        self.head_num = 8
        self.dropout_p = 0.5
        self.data_name= 'TSMC2014_NYC'
        self.learning_rate = 20 * 1e-4
        self.lr_step = 2
        self.lr_decay = 0.1
        self.L2 = 1 * 1e-5
        self.clip = 5.0
        self.epoch_max = 100
        self.data_path = '../data/'
        self.save_path = '../results/'
        ######
        self.vid_list = data['locID_dict']
        self.uid_list = data['userID_dict']
        self.data_neural = data['data_nrl']

        self.tim_size = 48
        self.loc_size = len(self.vid_list)
        self.uid_size = len(self.uid_list) 
        self.model_list =  ['rnn', 'context-attention', 'new-attention']
        self.model_type = self.model_list[4]

args = Setting()