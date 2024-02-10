from zenml.steps import BaseParameters

class ModelNameConfig(BaseParameters):
    """Model Configs"""
    model_name: str = "LogisticRegression"
    #model_kwargs: dict = {'C':1, 'multi_class':'ovr', 'solver':'lbfgs', 'random_state':42, 'max_iter':1000}
