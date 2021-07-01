# learning_control_solo

code for the paper
"Leveraging Forward Model Prediction Error for Learning Control"
 Sarah Bechtle, Bilal Hammoud, Akshara Rai,Franziska Meier and Ludovic Righetti. 

Abstract: 
Learning for model based control can be sample-efficient and generalize well, however successfully learning models and controllers that represent the problem at hand can be challenging for complex tasks. Using inaccurate models for learning can lead to sub-optimal solutions, that are unlikely to perform well in practice. In this work, we present a learning approach which iterates between model learning and data collection and leverages forward model prediction error for learning control. We show how using the controller's prediction as input to a forward model can create a differentiable connection between the controller and the model, allowing us to formulate a loss in the state space. This lets us include forward model prediction error during controller learning and we show that this creates a loss objective that significantly improves learning on different motor control tasks. We provide empirical and theoretical results that show the benefits of our method and present evaluations in simulation for learning control on a 7 DoF manipulator and an underactuated 12 DoF quadruped. We show that our approach successfully learns controllers for challenging motor control tasks involving contact switching.


Instructions:

To install the simulation environment please refer to https://github.com/open-dynamic-robot-initiative/robot_properties_solo 

dependencies: pytorch, numpy

To run the code do: python mbrl_solo.py

Description of the code:

The code implements the model based controller learning framework presented in the paper "Leveraging Forward Model Prediction Error for Learning Control". Where we present an unbiased loss function for controller learning, that trades off predictive task performance and forward model quality. This loss results in a controller that shifts the observed data distribution such that the collected data, when used to learn the models, reduces model bias, improves model quality and, as a consequence, improves task learning. In particular, we also showed how this unbiased loss can be successfully used on contact rich tasks like walking on the quadruped solo.

The code is structured as follows: mbrl_solo.py implements the model-based learning loop that alternates between forward model learning and policy learning. The model_module.py file contains the neural network models for the forward model and the controller, it also contains the training procedure. utils.py contains helper functions for data collection and learning. 

