class Critic():

    def __init__(self, method="TL", nn_dimensions=None, lr=0.5, eligibility_decay=0, discount_factor=0.5):
        self.lr = lr