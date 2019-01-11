from mlp_data_loader import BasicDataLoader
from rnn_data_loader import RnnDataLoader
        
def CreateDataLoader(opt):
    data_loader = None
    if opt.classifier_type == 'mlp':
        data_loader = BasicDataLoader()
    else:
        data_loader = RnnDataLoader()

    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader