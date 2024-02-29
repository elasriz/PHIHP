# PHIHP: Physics-Informed Model and Hybrid Planning for Efficient Dyna-Style Reinforcement Learning

## Installation

To run this project, you need to have the following prerequisites installed on your machine:
```
pip install -r requirements.txt
```

## Training a Physics-informed model

<p align="center">
  <img src='media/model_learning2.jpg' width="600"/>
</p>

```
python src/train.py task=dog-run
```


##  Training an model-free Actor-Critic agent (TD3) 

<p align="center">
  <img src='media/policy2.jpg' width="600"/>
</p>

```
python src/train.py task=dog-run
```

##  Hybrid Control 

<p align="center">
  <img src='media/agent3.jpg' width="600"/>
</p>

```
python src/train.py task=dog-run
```
