

def agent_factory(env, observer, agent_class: str, **params: dict):
    
    if agent_class == "CEMPlanner":
        from source.cross_entropy_method import CEMPlanner
        return CEMPlanner(env, observer, **params)
    
    elif agent_class == "RandomPolicy":
        from source.random_policy import RandomPolicy
        return RandomPolicy(env, **params)
    
    elif agent_class == "PHIHP":
        from source.PHIHP import PHIHP
        return PHIHP(env, observer, **params)

