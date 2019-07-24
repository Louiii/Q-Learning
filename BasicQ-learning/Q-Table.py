import numpy as np
import random
from Quiver import plotPolicy, makeUVM
from make_gif_from_png import makeGIF

startingState=(0,0)
goal=(4,4)
costs=[]
Q={}
actions=['up','right']
γ=0.9

#define a cost function across each state
for i in range(5):
    row=[]
    for j in range(5):
        ij = 5*(4+j-i)**2
        # if ij > 0:
        row.append( ij )
        # else:
        #     row.append(0)
        for action in actions:
            Q[((i,j),action)]=0
    costs.append(row)
costs = np.array(costs)


def nextState(state, action):# returns the new state, from doing an action in a state
    (i,j)=state
    if action=='up':
        if i==4:
            return state
        return (i+1,j)
    else:
        if j==4:
            return state
        return (i,j+1)

def immediateReward(state, action):# returns the cost of being in a state and performing an action
    if nextState(state, action)==goal:
        return 100
    (i,j)=state
    r = -costs[i][j]
    if action=='up':
        if i==4:
            r-=20
    else:
        if j==4:
            r-=20
    return r

def explore():# returns random action
    return random.choice(actions)

def maxAction(state):# returns the action with the highest Q-value
    Qs=[(action, Q[(state, action)]) for action in actions]
    actionQ= max(Qs, key=lambda x:x[1])
    return actionQ[0]

def experimentationStrategy(state, k):
    """ Chooses a greedy action (highest Q-value) or a random action using a pdf
        defined by the function: k ^ Q-value-of-action, (after normalisation).
    """
    normalisation = sum([k**q for key, q in Q.items() if key[0] == state ])
    probs = [( key, (k**q)/normalisation ) for key, q in Q.items() if key[0] == state ]
    r = random.random()# random.uniform(0, 1)
    accumulator = 0
    for (key, p) in probs:
        accumulator += p
        if accumulator >= r:
            return key[1] # the action

def episode():
    """ This is center of the algorithm, the agent gets put into a random state,
        then keeps performing actions until it gets to the goal state.

        The Q-table gets updated according to our recusive definition to
        appoximate the true Q-value.
    """
    state=(random.randint(0,4),random.randint(0,4))
    while(True):
        # action = explore()
        action = experimentationStrategy(state, 2)
        r = immediateReward(state, action)
        newState = nextState(state, action)
        Q[(state,action)] = r + γ*Q[(newState,maxAction(newState))]
        if newState == goal:
            break
        state = newState

def runNEpisodes(n, iterations_to_record):# runs n episodes and records at the list of iterations

    for iteration in range(1, n+1):
        episode()

        if iteration == iterations_to_record[0]:
            _ = iterations_to_record.pop(0)
            X, Y, U, V = makeUVM(Q)
            fn = "plots/Q"+str(iteration)+".png"
            title = "Policy visualisation. Iteration: "+str(iteration)
            plotPolicy(X, Y, U, V, costs, show=False, filename=fn, title=title, cbarlbl="Costs")

if __name__=="__main__":

    iterations_to_record = list( np.concatenate([np.arange(1,10,1),np.arange(10,30,2),np.arange(30,50,4),np.arange(50,100,10),np.arange(100,200,20),np.arange(200,500,40),np.arange(500,1001,50)]) )
    runNEpisodes(1000, iterations_to_record)
    makeGIF('Policy-ExperimentationStrategy', 0.25, 12)
    # print(Q)
