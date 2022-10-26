# Create a class for training results

class TrainingResults(object):
    '''
    This object makes storing and accessing training results easier.
    '''
    
    def __init__(self):
        '''
        Results are initialized as empty lists.
        '''
        self.names   = []
        self.loss_tr = []
        self.loss_va = []
        self.time_tr = []
        self.stop_model = None
        self.stop_epoch = 0
        self.stop = False
    
    def loadedModel(self, name):
        '''
        A model was loaded so just append the name and placeholders.
        '''
        self.names.append(name)
        self.loss_tr.append([])
        self.loss_va.append([])
        self.time_tr.append(0)
    
    def appendResults(self, results_tr):
        '''
        Append a separate TrainingResults object to this one. This is done when
        storing training results from multiple models in one object. Add the
        lists from the new results to the end of this object's results lists
        '''
        self.names.extend(results_tr.names)
        self.loss_tr.extend(results_tr.loss_tr)
        self.loss_va.extend(results_tr.loss_va)
        self.time_tr.extend(results_tr.time_tr)
    