import random
import numpy as np
import simpy
import globalvar

class Executor():

    def __init__(self, env, candidate, task):
        self.candidate = candidate
        self.env = env
        self.task = task

    def run(self):
        start = self.env.now
        req = self.candidate.queueResource.request()
        yield req
        waitTime = self.env.now - start         
        serviceTime = self.candidate.getServiceTime()
        yield self.env.timeout(serviceTime)
        self.candidate.queueResource.release(req)

        #monitor CPU utilization
        globalvar.c[self.candidate.getID()] = globalvar.c[self.candidate.getID()]+(self.env.now-start-waitTime)
        if self.env.now > globalvar.lengthOfInterval*globalvar.k:
            globalvar.cpu.append([round(i * (1/globalvar.lengthOfInterval),2) for i in globalvar.c])
            globalvar.c = [0 for x in range(globalvar.numberOfNodes)]
            globalvar.k = globalvar.k+1



        #monitor number of arrivals
        globalvar.q[self.candidate.getID()] = globalvar.q[self.candidate.getID()]+1
        if self.env.now > globalvar.lengthOfInterval*globalvar.j:
            globalvar.numberOfArrivals.append(globalvar.q)
            globalvar.q = [0 for x in range(globalvar.numberOfNodes)]
            globalvar.j = globalvar.j+1

        #continue to process task        
        self.candidate.schedule(self.env, self.task)
