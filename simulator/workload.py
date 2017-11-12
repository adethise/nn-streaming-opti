import random
import simpy
import task
import executor
import globalvar

class Workload():
  
    def __init__(self, env, id, numRequests):
        self.id = id
        self.env = env
        self.spoutList = []
        self.numRequests=numRequests
        env.process(self.run())
       

    def run(self):
        
        taskCounter = 0
        total = len(self.spoutList)
        while(self.numRequests!=0):

            taskToSchedule = task.Task("Task " + str(taskCounter), self.env)
            taskCounter = taskCounter+1

            spoutNode = makeChoice(total, self.spoutList)
            print("Workload %s sends out %s to %s at time %d" % (self.id, taskToSchedule.getID(), spoutNode.getName(), self.env.now))

            executorProcess = executor.Executor(self.env, spoutNode, taskToSchedule)
            self.env.process(executorProcess.run())
            
            yield self.env.timeout(10)
            #self.reduceRequests()

    def addSpout(self, a_spout):
        self.spoutList.append(a_spout)


    def getID(self):
        return self.id

    def reduceRequests(self):
        self.numRequests = self.numRequests-1
    

def makeChoice(total, spoutList):
    r = random.uniform(0, total)
    upto = 0
    for spout in spoutList:
        if upto + 1 > r:
            return spout
        upto += 1
    assert False, "Shouldn't get here"

