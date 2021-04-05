class Node:

    def __init__(self, state, action, player=None):
        self.state = state
        self.player = player
        self.action = action 
        self.parent = None
        self.children = []

        self.Q = 0
        self.N = 0

    def visit(self):
        self.N += 1
    
    def set_parent(self, parent):
        self.parent = parent

    def set_action(self, action):
        self.action = action

    def get_children(self):
        return self.children