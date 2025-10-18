#       __PLOTTINGTOOLS__
#       Plotting functions
#
#       Authors: 
#       Kostas Alexis (konstantinos.alexis@mavt.ethz.ch)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
    
def plot3(a, b, c, mark="o", col="r"):
    # mimic matlab plot3
    # Python 3.10 conversion - updated matplotlib usage
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.invert_zaxis()
    ax.invert_xaxis()
    ax.set_aspect('equal', 'datalim')
    ax.plot(a, b, c, color=col, marker=mark)
    plt.show()