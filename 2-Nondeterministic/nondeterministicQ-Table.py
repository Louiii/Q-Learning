import numpy as np
import random
from Quiver2 import *
from make_gif_from_png import makeGIF
#from statistics import stdev, mean
from mvn import norm_pdf_multivariate
from numpy import *


class QSample():
    def __init__(self):
        self.q=0
        self.visits=0

    def update(self, q, visits):
        self.q=q
        self.visits=visits

class RLAgent():

    def __init__(self, w, h, startingState, goal, actions, γ, means):
        # super(, self).__init__()
        self.Q = {}
        self.w, self.h = w, h
        self.startingState, self.goal = startingState, goal
        self.γ = γ
        self.actions = actions

        # init Q dictionary:
        for i in range(w):
            for j in range(h):
                for action in actions:
                    self.Q[((i,j),action)] = QSample()
        self.means = means
        self.costs_average=np.zeros(shape=(w,h))
        self.mask = self.newMask()
        self.states = []
        self.loggedRewards = []
        self.plotroot = "temp-plots"
        self.logroot = "reward-logs"
        n=500
        for i in range(w):
            for j in range(h):
                self.states.append( (i, j) )
                # costs_average[i, j] = sum( [cost((i,j), True) for _ in range(n)] )/n
                self.costs_average[i, j] = sum( [self.cost((i,j), True) for _ in range(n)] )/n

    def newMask(self):
        self.mask = [k for k in range(len(means)-1) if random.random() < 0.5]
        if random.random() < 0.05:
            self.mask.append( len(self.means) )
        # return self.mask

    def cost(self, state, nMask=False):
        if nMask:
            self.newMask()
        (i, j) = state
        cov = matrix([[0.3, 0], [0, 0.3]])
        c=sum([norm_pdf_multivariate(array([i, j]), array(self.means[k]), cov) for k in range(len(self.means)) if k in self.mask])
        return 100*c

    def nextState(self, state, action):# returns the new state, from doing an action in a state
        (i,j)=state
        if action=='right':
            if i==w-1:
                return state
            return (i+1,j)
        elif action=='up':
            if j==h-1:
                return state
            return (i,j+1)

    def immediateReward(self, state, action, average=False):# returns the cost of being in a state and performing an action
        if self.nextState(state, action)==self.goal:
            return 100
        (i,j)=state
        # r = -costs[i,j]
        if average:
            r = -self.costs_average[state]
        else:
            r = -self.cost(state)

        if action=='right':
            if i==w-1:
                r-=20
        elif action=='up':
            if j==h-1:
                r-=20
        return r

    def valueOfPolicy(self):
        total_r = 0
        for s in self.states:
            st = s
            r = -self.costs_average[st]
            while True:
                r += self.immediateReward(st, self.maxAction(st), True)
                if st == self.nextState(st, self.maxAction(st)):
                    break
                st = self.nextState(st, self.maxAction(st))
                if st == goal:
                    r+=100
                    break
            total_r += r
        return total_r



    def explore(self):# returns random action
        return random.choice(self.actions)

    def maxAction(self, state):# returns the action with the highest Q-value
        Qs=[(action, self.Q[(state, action)].q) for action in self.actions]
        actionQ= max(Qs, key=lambda x:x[1])
        return actionQ[0]

    def εGreedy(self, state, ε):
        if random.random() < ε:
            return self.explore()
        else:
            return self.maxAction(state)

    def softmax(self, state, τ, e=np.exp(1)):
        """ τ is a positive param called temp.
        high τ ~ equiprobable actions
        low τ ~ best actions most probable

        requires knowledge of Q-values/powers of e
        """
        qs={}
        for a in self.actions:
            qs[a] = self.Q[(state, a)].q# all the possible Q-values from our state
        sumQ = sum([ e**( qs[a]/τ ) for a in self.actions ])
        probs = [ (a, e**( qs[a]/τ )/sumQ ) for a in self.actions ]
        print(probs)
        r = random.random()
        accumulator = 0
        for (action, p) in probs:
            accumulator += p
            if accumulator >= r:
                return action

    # def experimentationStrategy(self, state, k):
    #     """ Chooses a greedy action (highest Q-value) or a random action using a pdf
    #         defined by the function: k ^ Q-value-of-action, (after normalisation).
    #     """
    #     qs={}
    #     for a in self.actions:
    #         qs[a] = self.Q[(state, a)].q# all the possible Q-values from our state
    #     # print(qs)
    #     l = list(qs.values())
    #     if all(x==l[0] for x in l):# If they are all the same, make a random choice
    #         return self.explore()
    #     # sd = stdev(l)
    #     mn = mean(l)
    #     for a in self.actions:
    #         qs[a] = qs[a]/mn#(qs[a]-mn)/sd + 2
    #     # print(qs)
    #     normalisation = sum([k**q for q in qs.values()])
    #     probs = [( action, k**q/normalisation ) for action, q in qs.items()]
    #     # print("probs: " +str(probs))
    #     r = random.random()
    #     accumulator = 0
    #     for (action, p) in probs:
    #         accumulator += p
    #         if accumulator >= r:
    #             return action

    def episode(self, τ):
        """ This is center of the algorithm, the agent gets put into a random state,
            then keeps performing actions until it gets to the goal state.

            The Q-table gets updated according to our recusive definition to
            appoximate the true Q-value.
        """
        self.newMask()
        state=(random.randint(0,w-1),random.randint(0,h-1))
        while(True):
            action = self.explore()
#            action = self.εGreedy(state, 0.2)
#            action = self.softmax(state, τ, 1.5)

            r = self.immediateReward(state, action)
            newState = self.nextState(state, action)
            # Update rule:
            # self.Q[(state,action)].q = r + self.γ*self.Q[(newState,self.maxAction(newState))].q
            visits = self.Q[(state,action)].visits + 1
#            prevQ = self.Q[(state,action)].q
            deterministicUpdate = r + self.γ*self.Q[(newState,self.maxAction(newState))].q
            α_n = 1/(1+visits)
            self.Q[(state,action)].q = (1 - α_n)*self.Q[(state,action)].q + α_n*deterministicUpdate
            self.Q[(state,action)].visits = visits
            if newState == self.goal:
                break
            state = newState

    def runNEpisodes(self, n, iterations_to_record):# runs n episodes and records at the list of iterations
        τ = 100

        for iteration in range(1, n+1):
            # print("iteration "+str(iteration))

            print(τ)
            self.episode(τ)
            τ*=0.9995

            # log total current reward, go from each starting point and follow policy
#            print("logging...")
            self.loggedRewards.append( ( iteration, self.valueOfPolicy() ) )#sum( [self.Q[(s, self.maxAction(s))].q for s in self.states] ) ) )
#            print("logging done.")

            if iteration == iterations_to_record[0]:
                _ = iterations_to_record.pop(0)
                X, Y, U, V = makeUVM(self.Q, self.w, self.h)
                fn = self.plotroot+"/Q"+str(iteration)+".png"
                title = "Policy visualisation. Iteration: "+str(iteration)
                plotPolicy(X, Y, U, V, self.costs_average, self.w, self.h, show=False, filename=fn, title=title, cbarlbl="Costs")

#if __name__=="__main__":
iterations_to_record = list( np.concatenate([np.arange(1,10,1),np.arange(10,30,2),np.arange(30,50,4),np.arange(50,100,10),
                                    np.arange(100,200,20),np.arange(200,500,40),np.arange(500,1000,50),np.arange(1000,2000,100),
                                    np.arange(2000,5000,250),np.arange(5000,10000,500),np.arange(10000,20000,1000),np.arange(20000,50001,5000)]) )
w, h = 6, 5
startingState=(0,0)
goal=(w-1,h-1)
Q={}
actions=['up','right']
γ=0.9
means = [[1, 1],[2, 2],[3, 3],[4, 3],[0, 4],[1, 4],[4, 0],[4, 1],[5, 0],[5, 1],[2, 3]]

agent = RLAgent(w, h, startingState, goal, actions,  γ, means)



#TRYING TO PLOT CTS COSTS
#def build_f(xs, ys):
#    f_list = []
#    mx, mn = -np.inf, np.inf
#    for x in xs:
#        row=[]
#        for [y] in ys:
#            c = agent.cost( (x, y) )
#            if c < mn:
#                mn=c
#            if c > mx:
#                mx=c
#            row.append( (x, y, c ) )
#        f_list.append(row)
#    return f_list, mx, mn
#
#
#xs = np.linspace(0, w, w*30)
#ys = np.linspace(0, h, h*30).reshape(-1, 1)
#fl, mx, mn = build_f(xs, ys)
#normalise = lambda x: x/mx #(x-mn)/(mx-mn)
#fl = [ [(r[0], r[1], normalise(r[2]) ) for r in row] for row in fl]
#plotCostCts(fl)


agent.runNEpisodes(5000, iterations_to_record)



name='random'
makeGIF(agent.plotroot, agent.logroot+"/"+name, 0.3, 15)

export_dataset(agent.loggedRewards, name)
plotRewardOverTime(agent.loggedRewards, "./"+agent.logroot+"/"+name+".png")


plotAllCosts(['random.csv', 'ε-Greedy.csv', 'softmax-tauchange.csv'], ['Random', 'ε-Greedy', 'Softmax'])#, τ=100, decay=0.9995, power=1.5'])

    # cost((0,0))
    # print("Q:")
    # print(agent.Q)
