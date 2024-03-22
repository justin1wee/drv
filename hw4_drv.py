"""
File: drv.py
Description: a file that contains DRV classes and functions
"""

import copy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class DRV:
    """ a model for discrete random variables where outcomes are numeric"""

    def __init__(self, dist=None, type='discrete', min=None, max=None,
                 mean=None, stdev=None, bins=10):
        self.type = type
        if dist is None:
            self.dist = {}  # outcome -> p(outcome)
            if type == 'uniform':
                self._init_uniform(min, max, bins)
            elif type == 'normal':
                self._init_normal(mean, stdev, bins)
        else:
            self.dist = copy.deepcopy(dist)

    def _init_uniform(self, min, max, bins):
        if min is None or max is None:
            raise ValueError("Uniform distribution requires 'min' and 'max' values.")

        # Generate outcomes and probability
        outcomes = np.linspace(min, max, bins)
        probability = 1 / bins
        self.dist = {outcome: probability for outcome in outcomes}

    def _init_normal(self, mean, stdev, bins):
        if mean is None or stdev is None:
            raise ValueError("Normal distribution requires 'mean' and 'stdev'.")

        # Generate values
        vals = np.random.normal(mean, stdev, 100)
        # Generate outcomes
        outcomes = np.linspace(min(vals), max(vals), bins)

        # Initiate distribution w/ counts
        for outcome in outcomes:
            self.dist[outcome] = 0
        # Generate counts
        for val in vals:
            bin = min(outcomes, key=lambda x: abs(x - val))
            self.dist[bin] += 1

        # Generate probabilities
        for x, c in sorted(self.dist.items()):
            self.dist[x] = c / len(vals)

    def __getitem__(self, x):
        return self.dist.get(x, 0.0)

    def __setitem__(self, x, p):
        self.dist[x] = p

    def apply(self, other, op):
        """ Apply a binary operator to self and other"""
        Z = DRV()
        items = self.dist.items()
        oitems = other.dist.items()
        for x, px in items:
            for y, py in oitems:
                Z[op(x, y)] += px * py
        return Z

    def applyscalar(self, a, op):
        Z = DRV()
        items = self.dist.items()
        for x, p in items:
            Z[op(x, a)] += p
        return Z

    def __add__(self, other):
        return self.apply(other, lambda x, y: x + y)

    def __radd__(self, a):
        return self.applyscalar(a, lambda x, c: c + x)

    def __rmul__(self, a):
        return self.applyscalar(a, lambda x, c: c * x)

    def __rsub__(self, a):
        return self.applyscalar(a, lambda x, c: c - x)

    def __sub__(self, other):
        return self.apply(other, lambda x, y: x - y)

    def __mul__(self, other):
        return self.apply(other, lambda x, y: x * y)

    def __truediv__(self, other):
        return self.apply(other, lambda x, y: x / y)

    def __pow__(self, other):
        return self.apply(other, lambda x, y: x ** y)

    def __repr__(self):
        xp = sorted(self.dist.items())
        rslt = ''
        for x, p in xp:
            rslt += str(x) + " : " + str(round(p, 8)) + " \n"

        return rslt

    def E(self):
        xp = sorted(self.dist.items())
        E = 0
        for x, p in xp:
            E += x*p

        return round(E, 3)

    def stdev(self):
        xp = sorted(self.dist.items())
        E = self.E()
        var = 0
        for x, p in xp:
            var += (((x - E) ** 2) * p)

        return np.sqrt(var)

    def random(self):
        outcomes = list(self.dist.keys())
        probs = list(self.dist.values())
        return np.random.choice(outcomes, p=probs)

    def plot(self, title='', xscale='', yscale='', show_c=False, trials=0, bins=10, width=1):
        """ Display the DRV distribution"""

        if trials == 0:
            plt.bar(self.dist.keys(), self.dist.values(), width=width)
            plt.ylabel('Probability p(x)')

            if show_c:
                c_values = []
                c_sum = 0
                for val in self.dist.values():
                    c_sum += val
                    c_values.append(c_sum)

                plt.plot(self.dist.keys(), c_values)
                plt.ylabel('Probability p(x)')

        else:
            sample = [self.random() for i in range(trials)]
            sns.displot(sample, kind='hist', stat='probability', bins=bins)

        plt.title(title)



        if yscale == 'log':
            plt.yscale('log')
            plt.ylabel('Log-scaled Probability')

        plt.xlabel('Value x')
        plt.show()


def main():

    # star formation rate
    # uniform distribution with minimum value of 1.5 and maximum value of 3.0
    rstar = DRV(type='uniform', min=1.5, max=3.0)

    # fraction of stars that have planets
    # discrete random variable with 94% probability of outcome being 1.0 and 6% being
    # lower than 1.0, but no probability of outcome being above 1.0
    fp = DRV(type='discrete', dist={1.0: 0.94, 0.99: 0.02, 0.98: 0.01, 0.97: 0.02, 0.96: 0.01})

    # of stars having planets, how many planets can support life
    # uniform distribution with minimum value of 1.0 and maximum value of 5.0
    n_e = DRV(type='uniform', min=1.0, max=5.0)

    # fraction of these life supporting planets that can develop life
    """ According to https://askanearthspacescientist.asu.edu/drake-equation#:~:text=The%20Drake%20Equation%20is%20part,easy%20to%20discover%20alien%20life.
    if life can exist, then it will (aka 100% or 1.0 probability) - this makes sense as it seems logical to assume
    that any planet with the capability of supporting life will have some form of life eventually appear
    
    A discrete distribution with one outcome (1.0) and one probability (1.0) seems to display this theory the best, 
    with the only outcome being that if a planet can support life, then life will eventually grow in that planet
    """
    fl = DRV(type='discrete', dist={1.0: 1.0})

    # fraction of planets with life that develop intelligent life
    """ According to the same source, although life will form on these planets, only a very small percentage of life
    will be intelligent life. The estimate for fi on the article was 1%, showing how rare it is for intelligent life to 
    form even with all the conditions for life on a planet being met
    
    A normal distribution with a mean of 0.03 and a stdev of 0.001 is what I decided on, due to how rare it is for these
    life forms to grow. However, I chose to increase the mean by 0.02 compared to the website because I believe that 
    there is life out there, and I don't think 0.01 is enough to justify other forms of intelligent life in space
    """
    fi = DRV(type='normal', mean=0.03, stdev=0.001)

    # fraction of intelligent life-bearing planets that develop technology that releases detectable signals into space
    """ According to the same source, intelligent life will most likely be able to create some form of communication
    with other planets if they were to exist. Specifically, they gave 50% as the chance of this happening. This makes 
    sense to me, as I have seen humans do it and wouldn't bet against another advanced civilization doing this as well.
    
    A normal distribution with a mean of 0.5 and a stdev of 0.1 makes sense to me in terms of the probability 
    distribution of this happening. This is because we could say about every one in two intelligent life forms will 
    create some sort of signal into space, while it is still unclear how accurate this estimate is (hence the stdev)
    """
    fc = DRV(type='normal', mean=0.5, stdev=0.1)

    # length of time (in years) during which these advanced civilizations will send out signals into space
    """ According to the website, intelligent life will last for 1,000,000 years. To me, this estimate sounds
    a bit high. I would probably estimate the length of time to be around 100,000 years; this is because even though 
    humans have been alive for millions of years, they did not become 'advanced' and capable of sending signals to 
    space until very recently.
    
    A normal distribution with mean of 100000 and stdev of 10000 seems to be a good estimate of how long advanced 
    civilizations last based on the estimates given.
    """
    L = DRV(type='normal', mean=100000, stdev=10000)

    # number of advanced civilizations in our galaxy that we might communicate with
    N = rstar * fp * n_e * fl * fi * fc * L
    print(N.E())
    print(N.stdev())

    N.plot(trials=100)

if __name__ == '__main__':
    main()