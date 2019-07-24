<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
# Q-Learning
Q-learning algorithms, inspired by Tom Mitchell - "Machine Learning" reinforcement learning chapter

# Basic Q-Learning Algorithm
## Description
### Gridworld
The valid states in this simple world are $$(i, j)$$ where $$i, j \in \Z \cap [0, 4]$$.

### Agent

## Structure
.
 * [BasicQ-learning](./BasicQ-learning)
   * [Q-Table.py](./BasicQ-learning/Q-Table.py)
   * [Quiver.py](./BasicQ-learning/Quiver.py)
   * [make_gif_from_png.py](./BasicQ-learning/make_gif_from_png.py)

## Running
```bash
$ python Q-Table.py
```
The model will train and output png files into the plots folder of its progress, then it will join them into a gif.

<!---
![](/BasicQ-learning/Policy-RandomExploring.gif) ![](/BasicQ-learning/Policy-ExperimentationStrategy.gif)
-->

## Policy Generation
Random Exploring           |  Greedy & Random Strategy
:-------------------------:|:-------------------------:
<img src="/BasicQ-learning/Policy-RandomExploring.gif" width="425"/> | <img src="/BasicQ-learning/Policy-ExperimentationStrategy.gif" width="425"/>

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

<!---
## License
[MIT](https://choosealicense.com/licenses/mit/)
-->
