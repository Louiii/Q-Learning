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
- The agents main feature is a Q-table, which maps the states and actions to a Q-value. The Q-values can be used to determine which action to take from a given state.
- Episode loop:
  - Select an action, a, according to a rule.
  - Update rule: Q(s,a) <- r(s, a) + gamma * max_{a'}( Q(s', a') )
  where s is the current state, a the choose action, r is the immediate reward, gamma is the decay constant, s' is the next state after performing action a in state s, and a' are the actions available from state s'.
  - Strategies for learning the optimal policy:
    - 1 Random action selection. This can explore the whole search space.
    - 2 Greedy action selection, the highest Q-value availiable from the current state. This can't explore the whole search space.
    - 3 Choose a greedy action or a random action using a pdf defined by the function: k^Q-value-of-action, (after normalisation). This can explore the whole search space, but chooses more promising actions with higher probability.

## File Structure
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

The colours of the arrows indicate the cost from being in a cell. The optimal policy indicates that we should move to cells with lower costs on our way to the goal, as expected. Notice how Greedy & Random Strategy settles on the optimal policy significantly faster than Random Exploring.

# 2 Nondeterministic Q-Learning Algorithm
# 3 Continous Q-Learning Algorithm
# 4 Nondeterministic and Continous Q-Learning Algorithm


<!---
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
-->
