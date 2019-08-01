import numpy as np
import random
from Quiver import plotPolicy, makeUVM
from make_gif_from_png import makeGIF


def nextState(state, action):# returns the new state, from doing an action in a state
    (i,j)=state
    if action=='right':
        if i==w-1:
            return state
        return (i+1,j)
    elif action=='up':
        if j==h-1:
            return state
        return (i,j+1)

def immediateReward(state, action):# returns the cost of being in a state and performing an action
    if nextState(state, action)==goal:
        return 100
    (i,j)=state
    r = -costs[i,j]
    if action == 'right':
        if i==w-1:
            r-=20
    elif action == 'up':
        if j==h-1:
            r-=20
    return r

def maxAction(state):# returns the action with the highest Q-value
    Qs=[(action, Q[(state, action)]) for action in actions]
    actionQ= max(Qs, key=lambda x:x[1])
    return actionQ[0]

def explore():# returns random action
    return random.choice(actions)

def εGreedy(state, ε):
    if random.random() < ε:
        return explore()
    else:
        return maxAction(state)

def softmax(state, τ, e=np.exp(1)):
    """ τ is a positive param called temp.
    high τ ~ equiprobable actions
    low τ ~ best actions most probable
    using a pdf defined by the function: k ^ Q-value-of-action, (after normalisation).
    requires knowledge of Q-values/powers of e
    """
    qs={}
    for a in actions:
        qs[a] = Q[(state, a)]# all the possible Q-values from our state
    sumQ = sum([ e**( qs[a]/τ ) for a in actions ])
    probs = [ (a, e**( qs[a]/τ )/sumQ ) for a in actions ]
    print(probs)
    r = random.random()
    accumulator = 0
    for (action, p) in probs:
        accumulator += p
        if accumulator >= r:
            return action

def episode(τ):
    """ This is center of the algorithm, the agent gets put into a random state,
        then keeps performing actions until it gets to the goal state.

        The Q-table gets updated according to our recusive definition to
        appoximate the true Q-value.
    """
    state=(random.randint(0,w-1),random.randint(0,h-1))
    while(True):
        # action = explore()
        # action = εGreedy(state, 0.2)
        action = softmax(state, τ, 1.5)

        r = immediateReward(state, action)
        newState = nextState(state, action)
        Q[(state,action)] = r + γ*Q[(newState,maxAction(newState))]
        if newState == goal:
            break
        state = newState

def runNEpisodes(n, iterations_to_record, strategy):# runs n episodes and records at the list of iterations
    τ = 100

    for iteration in range(1, n+1):
        episode(τ)
        τ*=0.999
        print(τ)

        if iteration == iterations_to_record[0]:
            _ = iterations_to_record.pop(0)
            X, Y, U, V = makeUVM(Q, w, h)
            fn = "temp-plots/Q"+str(iteration)+".png"
            title = "Policy, "+strategy+" strategy. Iteration: "+str(iteration)
            plotPolicy(X, Y, U, V, costs, w, h, show=False, filename=fn, title=title, cbarlbl="Costs")
            # plotPolicy(X, Y, U, V, costs, w, h, show=False, filename=fn)


w, h = 5, 5
startingState=(0,0)
goal=(w-1,h-1)
costs=[]
Q={}
actions=['up','right']
γ=0.9

# define a cost function across each state
# first cost function:
for i in range(w):
    row=[]
    for j in range(h):
        ij = 5*(4+j-i)**2
        # if ij > 0:
        row.append( ij )
        # else:
        #     row.append(0)
        for action in actions:
            Q[((i,j),action)]=0
    costs.append(row)
costs = np.array(costs)


iterations_to_record = list( np.concatenate([np.arange(1,10,1),np.arange(10,30,2),np.arange(30,50,4),
    np.arange(50,100,10),np.arange(100,200,20),np.arange(200,500,40),np.arange(500,1000,50),np.arange(1000,2001,200)]) )
strategy='softmax'
runNEpisodes(2000, iterations_to_record, strategy)
makeGIF('temp-plots', './gifs/'+strategy, 0.25, 12)# framerate=0.25s, 12 repeats at the end
