from collections import defaultdict

class CFG():
    
    def __init__(self):
        self.rules = defaultdict(lambda: defaultdict(int))
  
      
    def add_child(self, child):
        self.children.append(child)
    
    def get_downward_arcs(self):
        return self._get_arcs()
        
    def _get_arcs(self):
        arcs = []
        for child in self.children:
            arc = (self.tag, child.tag)
            arcs.append(arc)
            arcs.extend(child._get_arcs())

        return arcs
            
        
        