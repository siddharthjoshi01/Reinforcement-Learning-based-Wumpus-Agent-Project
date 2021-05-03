import gym
import gym_wumpusworld
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import csv

def allStates(env):
    states = []



    directions = ['NORTH','EAST','SOUTH','WEST']

    for i in range(env.size):
        for j in range(env.size):
            for direction in directions:
                states.append((i,j,direction))
    return states

def legalActions(state,nextStep):
    #print(nextStep[0])
    
    actions = [0,1,2,3,4,5]

    if not(nextStep[0]['gold'] and nextStep[0]['x'] == 0 and nextStep[0]['y'] == 0):
        actions.remove(5)  
    if nextStep[0]['gold']:
        actions.remove(3)
    if not nextStep[0]['arrow']:
        actions.remove(4)   
        
    return actions

def selectActionE(state, qTable, nTable, nextStep, epsilon):
    # Policy
    
    actions = legalActions(state,nextStep)
    eTable = np.zeros(6)

    for a in actions:
        eTable[a] = expFunc(qTable[state][a], nTable[state][a])
      
    action = np.argmax(eTable)

    return action

def getEpsilon(epsilon, minE, eRate):

    if epsilon > 3*minE:
        epsil = epsilon/(1 + eRate)
    else:
        epsil = 0
    return epsil

def getAlpha(alpha, minA, aRate):

    if alpha >= minA:

        alph = alpha - aRate
    else:
        alph = minA
    return alph

def agentAlive(env,nextStep):
    
    pitLoc = []
    for i in env.pitLoc:
        pitLoc.append((i.x,i.y))
    # print(pitLoc)
    # print((env.obs['x'],env.obs['y']))
    # print((env.wumpLoc.x,env.wumpLoc.y))
    
    if (nextStep[0]['x'],nextStep[0]['y']) in pitLoc:
        return False
    elif nextStep[0]['wumpus_alive']:
        if (nextStep[0]['x'],nextStep[0]['y']) == (env.wumpLoc.x,env.wumpLoc.y):
            return False
        else:
            return True
    else:
        return True

def csvWriter(i,score,fieldnames):

    with open('score.csv', 'a') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            info = {
                "episode": i,
                "score": score
            }

            csv_writer.writerow(info)

def expFunc(u,n):
    k = 1000
    expFunc_val = u + (k/n)
    
    return expFunc_val

if __name__ == "__main__":

    env = gym.make('Wumpus-v0')
    qTable = np.zeros((64,6))
    nTable = np.ones(qTable.shape)

    num_episodes = 20000
    
    # Hyper parameters
    gamma = 0.5

    maxE = 0.5
    minE = 0.02
    eRate = 0.01

    maxA = 1
    minA = 0.01
    aRate = 0.1

    
    alpha = maxA

    fieldnames = ["episode", "score"]


    with open('score.csv', 'w') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
    
    score = 0
    i = 0
    while(i<num_episodes):
        
        nTable = np.ones(qTable.shape)
        env.reset()
               
        epsilon = maxE
        states = allStates(env)

        nextStep = [env.obs,1]
        timestep = 0
        csvWriter(i,score,fieldnames)
        
        print("episode: ", i)
        
        while(timestep<=490):
            if(agentAlive(env,nextStep)):

                epsilon = getEpsilon(epsilon, minE, eRate)

                direc = str(nextStep[0]['direction'])
                direction = direc.replace('Direction.','')
                
                state = states.index((nextStep[0]['x'],nextStep[0]['y'],direction))
                action = selectActionE(state, qTable, nTable, nextStep, epsilon)
                nTable[state][action] += 1

                nextStep = env.step(action)
                
                sPrime = states.index((nextStep[0]['x'],nextStep[0]['y'],direction))
                reward = nextStep[0]['score']

                futureActions = [0,1,2,3,4,5]         
                
                qnDummy = []

                for aPrime in futureActions:
                    expFunc_val = expFunc(qTable[sPrime][aPrime],nTable[sPrime][aPrime])
                    qnDummy.append(expFunc_val)
                
                maxQ = max(qnDummy)

                qUpdate = reward + (gamma * maxQ)
                
                
                qTable[state][action] = ((1-alpha) * qTable[state][action]) +  ( alpha * qUpdate )

               
                                
                score = nextStep[0]['score']
                
                if action == 5:
                    break
               
                timestep += 1
                env.render()
                # time.sleep(0.1)
            
            else:
                break
            
            i += 1

                

            
            
            

                
                
                

                







        