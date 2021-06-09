from egtplot import plot_static
from egtplot import load_bomze_payoffs
from egtplot import plot_animated
from moviepy.video.VideoClip import VideoClip

import urllib.request
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')


############################################# Picture version #############################################

def init_params():
    return [[1], [1], [1], [1]], ['S', 'D', 'I']

def get_payoff(alpha, beta, gamma, rho):
    return [[0, alpha, 0], 
            [1 + alpha - beta, 1 - 2 * beta, 1 - beta + rho], 
            [1 - gamma, 1 - gamma, 1 - gamma]]

def test1():
    payoff_entries = [[0], [-1], [3], [-1], [0], [1], [3], [1], [0]]
    simplex = plot_static(payoff_entries)
    plt.show()

def test2():
    payoffs = load_bomze_payoffs()
    payoff_entries = payoffs[34]
    print(payoff_entries)

def test3():
    parameter_values, labels = init_params()
    simplex = plot_static(parameter_values, custom_func=get_payoff, vert_labels=labels)
    plt.show()

def test4():
    parameter_values, labels = init_params()
    simplex = plot_static(parameter_values, custom_func=get_payoff, vert_labels=labels, paths=True)
    plt.show()

def test5():
    parameter_values, labels = init_params()
    simplex = plot_static(parameter_values, custom_func=get_payoff, vert_labels=labels, background=True)
    plt.show()

def test6():
    parameter_values, labels = init_params()
    simplex = plot_static(parameter_values, custom_func=get_payoff, vert_labels=labels,
                            paths=True, 
                            generations=10,
                            steps=2000,
                            ic_type='random',
                            path_color='viridis',
                            eq=False)
    plt.show()

def test7():
    parameter_values = [[1, 2], [1], [1], [1]]
    _, labels = init_params()
    simplex = plot_static(parameter_values, custom_func=get_payoff, vert_labels=labels)
    plt.show()

def test8():
    parameter_values = [[1, 2], [1, 3], [1], [1]]
    _, labels = init_params()
    simplex = plot_static(parameter_values, custom_func=get_payoff, vert_labels=labels, background=True)
    plt.show()

def test9():
    parameter_values = [[1, 2], [1, 3], [1, 4], [1]]
    _, labels = init_params()
    simplex = plot_static(parameter_values, custom_func=get_payoff, vert_labels=labels,
                            steps=600,
                            paths=True,
                            ic_type='edge',
                            ic_num=20)
    plt.show()

def test10():
    parameter_values = [[1, 2], [1, 3], [1, 4], [1]]
    _, labels = init_params()
    simplex = plot_static(parameter_values, custom_func=get_payoff, vert_labels=labels,
                            steps=2000,
                            background=False, 
                            ic_type='random',
                            paths=True,
                            path_color='hot',
                            eq=False)
    plt.show() 

############################################# Animated version #############################################

def test11():
    parameter_values =  [ [1],[2],[3], [4],[5],[6], [7],[7],[7] ]
    animation, fps = plot_animated(parameter_values)
    print(fps)

    # payoff = [ [1],[2],[3], [4],[5],[6], [7],[7],[7] ]
    # output = plot_animated( payoff )
    # print(type(animation))
    # plt.show()
    animation.write_videofile("my_animation.mp4", fps=fps)
    # animation.write_gif('animation_1.gif', program='imageio', loop=1, fps=fps) 

test11()