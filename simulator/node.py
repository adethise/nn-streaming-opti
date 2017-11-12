import random
import simpy
import executor
import experiment
import globalvar


class Node():
    def __init__(self, env, name, ID, is_Spout, is_End):
        self.env = env
        self.name = name
        self.ID = ID
        self.is_Spout = is_Spout
        self.is_End = is_End
        self.neighbors = []
        self.serviceTime = 5
        self.request_duration = 3
        self.queueResource = simpy.Resource(env, capacity=1)
        env.process(self.run())

    def run(self):
        
        random.seed()
        yield self.env.timeout(1)
       

    def addNeighbor(self, neighbor):
        self.neighbors.append(neighbor)


    def getName(self):
        return self.name

    def getID(self):
        return self.ID

    def getServiceTime(self):
        return self.serviceTime

    def schedule(self, env, taskToSchedule):
        
        #monitor throughput
        if env.now > globalvar.lengthOfInterval*globalvar.i:
                globalvar.throughput.append(globalvar.t)
                globalvar.t = 0
                globalvar.i = globalvar.i + 1
        if (self.is_End):
            taskToSchedule.sigTaskComplete()
            print("%s is a final node" % self.name) 
            globalvar.t = globalvar.t+1
        else:
            print(print("%s is not a final node" % self.name))
            
            self.sendRequest(taskToSchedule)
  

    def sendRequest(self, taskToSchedule):
        
        delay = self.request_duration

        #choose destionation
        i = round(random.uniform(0, len(self.neighbors)))-1
        candidate = self.neighbors[i]
        task = taskToSchedule

        print("%s sends %s to %s at time %d" % (self.name, taskToSchedule.id_, candidate.name, self.env.now))
        messageDeliveryProcess = DeliverMessageWithDelay(self.env, task, candidate, delay)
        self.env.process(messageDeliveryProcess.run(taskToSchedule, delay))


    def enqeueTask(self, taskToSchedule, candidate):
        executorProcess = executor.Executor(self.env, candidate, taskToSchedule)
        self.env.process(executorProcess.run())





 
class DeliverMessageWithDelay():
    def __init__(self, env, task, candidate, delay):
        self.env = env
        self.candidate = candidate
        self.delay = delay
    

    def run(self, taskToSchedule, delay):
        yield self.env.timeout(delay)
        self.candidate.enqeueTask(taskToSchedule, self.candidate)


