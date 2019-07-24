# Q-Learning Algorithms
Q-learning algorithms, inspired by Tom Mitchell - "Machine Learning" reinforcement learning chapter

# 1 Basic Q-Learning Algorithm
## Description
### Gridworld (Discrete)
- The valid states in this simple world are (i, j) where i, j are integers between 0 and 4, representing a 5x5 grid. 
- The valid actions are 'up' or 'right'.
- There are different costs associated with each cell according to a cost function, the goal always has a reward of 100.
- If the agent hits a wall it get a penalty but doesn't move anywhere.
- The world is deterministic.
- The goal state is (4, 4) in the top right corner, so starting at any state the agent can reach the goal.
- An episode is the sequence of actions and rewards from a random starting position until it reaches the goal.

### Agent
- The agents main feature is a Q-table.
- 

## Structure
 * [BasicQ-learning](./BasicQ-learning)
   * [Q-Table.py](./BasicQ-learning/Q-Table.py) -- contains all the code for the algorithm.
   * [Quiver.py](./BasicQ-learning/Quiver.py) -- visualising the policy.
   * [make_gif_from_png.py](./BasicQ-learning/make_gif_from_png.py) -- making the gif.
   * [/plots](./BasicQ-learning/plots) -- folder for the images.

## Running
```bash
$ python Q-Table.py
```
The model will train and output png files of its progress into the 'plots' folder, then it will join them into a gif.

## Policy Generation
Random Exploring           |  Greedy & Random Strategy
:-------------------------:|:-------------------------:
<img src="/BasicQ-learning/Policy-RandomExploring.gif" width="425"/> | <img src="/BasicQ-learning/Policy-ExperimentationStrategy.gif" width="425"/>

# 2 Nondeterministic Q-Learning Algorithm
# 3 Continous Q-Learning Algorithm
# 4 Nondeterministic and Continous Q-Learning Algorithm


<!---
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
-->
