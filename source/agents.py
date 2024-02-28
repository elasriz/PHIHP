from source.cross_entropy_method import CEMPlanner
from source.random_policy import RandomPolicy


def agent_factory(env, observer, agent_class: str, **params: dict):
    
    if agent_class == "CEMPlanner":
        return CEMPlanner(env, observer, **params)
    
    elif agent_class == "RandomPolicy":
        return RandomPolicy(env, **params)
    
    elif agent_class == "PICEM":
        from source.PICEM import PICEM
        return PICEM(env, observer, **params)

    elif agent_class == "PICEM_train":
        from source.PICEM_train import PICEM
        return PICEM(env, observer, **params)