# python objects to store oponent information

class Oponent:
    def __init__(self, result = 0, finalUtility = 0, offerVariance = []):
        self.result = result
        self.finalUtility = finalUtility
        self.offerVariance = offerVariance



def get_opponent_data(name):
    #for now returns default opponent
    henri = Oponent()
    #also can be henri = Oponent(1, .5, [1,2,2,,1])
    #you can also do henri.result [or whatever you name the return of this function] and compute on that
    return henri