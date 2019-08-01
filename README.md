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
    - 2 ε-Greedy action selection, the highest Q-value availiable from the current state, or random action witht probability ε.
    - 3 Softmax; Choose actions using a pdf defined by the function: k^Q-value-of-action, (after normalisation). This can explore the whole search space, but chooses more promising actions with higher probability.

## File Structure
 * [BasicQ-learning](./BasicQ-learning)
   * [Q-Table.py](./BasicQ-learning/Q-Table.py) -- contains all the code for the algorithm.
   * [Quiver.py](./BasicQ-learning/Quiver.py) -- visualising the policy.
   * [make_gif_from_png.py](./BasicQ-learning/make_gif_from_png.py) -- making the gif.
   * [/temp-plots](./BasicQ-learning/temp-plots) -- folder for the images.
   * [/gifs](./BasicQ-learning/gifs) -- folder for the gifs.

## Running
```bash
$ python Q-Table.py
```
The model will train and output png files of its progress into the 'plots' folder, then it will join them into a gif.
In the code you can switch between Random exploring, ε-Greedy and Softmax functions to determine how it the exploration strategy. Two lines must be changes to do this: the first line in the while loop in the function 'episode()', and the variable 'strategy' near the end of the file.

## Policy Generation
<!-- 
Random Exploring           |  Greedy & Random Strategy (k=2)
:-------------------------:|:-------------------------:
<img src="/BasicQ-learning/Policy-RandomExploring.gif" width="425"/> | <img src="/BasicQ-learning/Policy-ExperimentationStrategy.gif" width="425"/> 
-->
Random Exploring           |  ε-Greedy, ε = 0.2        |  Softmax, τ = 100, decay = 0.999
:-------------------------:|:-------------------------:|:-------------------------:
<img src="/BasicQ-learning/gifs/Random.gif" width="350"/> | <img src="/BasicQ-learning/gifs/εGreedy.gif" width="350"/> | <img src="/BasicQ-learning/gifs/softmax.gif" width="350"/>

The colours of the arrows indicate the cost from being in a cell. The optimal policy indicates that we should move to cells with lower costs on our way to the goal, as expected. Notice how Greedy & Random Strategy settles on the optimal policy significantly faster than Random Exploring.

### Different Cost Function
Random Exploring           |  Greedy & Random Strategy (k=2)
:-------------------------:|:-------------------------:
<img src="/BasicQ-learning/newCostFnRandomExplore.gif" width="425"/> | <img src="/BasicQ-learning/newCostFnExperimentalStrategy.gif" width="425"/>



Notice that the random strategy settles on an optimal solution after about 1000 iterations, however the Greedy & Random Strategy (with k=2) doesn't settle on an optimal strategy even after 50000 iterations. This is because of the Q-values being so large, the probabilities are generated from 2^{large numbers} which results in one action being chosen with probability ~1 and the rest ~0, without ensuring the action is in fact optimal.
This requires some sort of normalisation of these Q-Values, I tried different methods such as:...


# 2 Nondeterministic Q-Learning Algorithm
## Description
### Gridworld (Discrete, Non-determinitic rewards)
- Everything as before, except now the costs are built from multivariate normal pdfs with means positioned around the board, a mask is used to determine which ones are used on any particular episode. This means the agent will now have to sample.

### Agent
- The agent has been restructure to be object oriented, and a separate class for the elements in the Q-table, to keep track of the current Q-value and how many times that state has been visited yet.
- Episode loop:
  - Select an action, a, according to a rule.
  - Update rule: Q(s,a) <- (1-α_n)*Q(s,a) + α_n*(r(s, a) + gamma * max_{a'}( Q(s', a') ))
  where α_n = 1/(1+visits(s, a)).

## File Structure
 * [NondeterministicQ-Learning](./NondeterministicQ-Learning)
   * [nondeterministicQ-Table.py](./NondeterministicQ-Learning/Q-Table.py) -- contains all the code for the algorithm.
   * [Quiver2.py](./NondeterministicQ-Learning/Quiver2.py) -- visualising the policy.
   * [mvn.py](./NondeterministicQ-Learning/mvn.py) -- pdf for multivariate normal dist.
   * [make_gif_from_png.py](./NondeterministicQ-Learning/make_gif_from_png.py) -- making the gif.
   * [/temp-plots](./NondeterministicQ-Learning/temp-plots) -- folder for the images.
   * [/gifs](./NondeterministicQ-Learning/gifs) -- folder for the gifs.
   * [/reward-logs](./NondeterministicQ-Learning/reward-logs) -- folder for the datasets of the total reward at each iteration.

## Running
```bash
$ python nondeterministicQ-Table.py
```
<img src="/NondeterministicQ-Learning/LearningRates.png" width="650"/>

# 3 Continous Q-Learning Algorithm
//notes to self:
- Make continuous visualisation of cost fn.
- Make state space cts.
- Make action space cts.

# 4 Nondeterministic and Continous Q-Learning Algorithm


<!---
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
-->
