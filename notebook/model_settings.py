import os

class ModelSettings:
    '''
    Settings for the model training and testing module.
    '''
    SEED = int(os.getenv("SEED", None))
    MODEL = os.getenv("MODEL", 'RandomForestClassifier')
    SCORING = os.getenv("SCORING", 'accuracy')
    RUN_EVALUATION = os.getenv("RUN_EVALUATION", "1")
