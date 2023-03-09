import numpy as np

class Frame:
    def __init__(self, energies, imp, counts, hist_width, bins):
        self.total_energy, self.kinetic_energy = energies
        self.imp = imp
        self.counts = counts
        self.hist_width = hist_width
        self.bins = bins
        
    def get_distribution(self):
        vs = [self.hist_width * (0.5 + i) for i in range(self.bins)]
        return np.array(vs), np.array(self.counts)
        

class Data:
    def __init__(self, path):
        self.frames = []
        
        with open(path) as f:
            self.steps = int(f.readline())
            self.skip = int(f.readline())
            self.n = int(f.readline())
            self.dt = float(f.readline())
            self.factor = float(f.readline())
            self.epsilon = float(f.readline())
            self.sigma = float(f.readline()) / self.factor
            self.velocity = float(f.readline()) / self.factor
            self.mass = float(f.readline())
            self.size = float(f.readline()) / self.factor
            self.max_velocity = float(f.readline()) / self.factor
            self.hist_width = float(f.readline()) / self.factor
            self.bins = int(f.readline())
            self.k = 1.380649e-23
            
            while True:
                f.readline()
                line = f.readline()
                
                if line == "":
                    break
                
                energies = tuple(map(lambda x: float(x) / self.factor ** 2, line.split()))
                line = f.readline()
                imp = tuple(map(lambda x: float(x) / self.factor, line.split()))
                counts = tuple(map(int, f.readline().split()))
                
                self.frames.append(Frame(energies, imp, counts, self.hist_width, self.bins))
                self.frames[-1].temperature = energies[1] / self.n / 1.5 / self.k
                
    def get_total_energies(self):
        return [i.total_energy for i in self.frames]
    
    def get_kinetic_energies(self):
        return [i.kinetic_energy for i in self.frames]
    
    def maxwell(self, v, T):
        return 4 * np.pi * v ** 2 * (self.mass / 2 / np.pi / self.k / T) ** 1.5 *\
            np.exp(-self.mass * v ** 2 / 2 / self.k / T)
