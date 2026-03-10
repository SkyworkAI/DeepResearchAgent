"""Knowledge Capsule for Research Agent"""
from enum import Enum
from typing import Dict, List
import time, uuid

class Phase(Enum): SPROUT="sprout"; GREEN="green_leaf"; YELLOW="yellow_leaf"; RED="red_leaf"; SOIL="soil"

class Capsule:
    def __init__(self, c, s, p="P2"):
        self.id=uuid.uuid4().hex[:8]; self.content=c; self.source=s; self.priority=p
        self.confidence=0.7; self.phase=Phase.SPROUT; self.created=time.time()
    def boost(self):
        self.confidence=min(1.0,self.confidence+0.03)
        self.phase=Phase.GREEN if self.confidence>=0.8 else Phase.SPROUT

class ResearchTracker:
    def __init__(self): self.capsules: Dict[str,Capsule]={}
    def add(self,c,s,p="P2"): k=Capsule(c,s,p); self.capsules[k.id]=k; return k
    def access(self,i): return bool(self.capsules.get(i) and self.capsules[i].boost())
    def high_conf(self,t=0.8): return [c for c in self.capsules.values() if c.confidence>=t]
