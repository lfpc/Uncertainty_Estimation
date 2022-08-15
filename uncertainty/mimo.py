from NN_models.mimo import MIMO_Ensemble as __MIMO
from uncertainty import ensemble


def MIMO_ensemble(model_class, num_classes, n_ensembles: int = 3, name='MIMO', softmax='log',*args):
    return ensemble.Ensemble(__MIMO(model_class, num_classes, n_ensembles, name, softmax, *args))