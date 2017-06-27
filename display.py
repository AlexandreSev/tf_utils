# coding: utf-8
import matplotlib.pyplot as plt
import seaborn as sns

def plot_callback(dico_callback, xlabel=None, ylabel=None, xlim=None, ylim=None, title=None, 
    saving=False, save_path="./plot.png", legend=False, show=True):
    """
    Create a plot with seaborn.

    Args:
        dico_callback (dictionary): A dictionary whith the different element to plot on the same graph
        xlabel (String): The label of x axis. Can be None.
        ylabel (String): The label of y axis. Can be None.
        xlim (2-uple(number)): Scaling of x axis.
        ylim (2-uple(number)): Scaling of y axis.
        title (String): Title of the plot.
        saving (Boolean): If true, the plot will be saved in save_path.
        save_path (String): The path where to save the plot, is saving==True.
        legend (Boolean): Show ot not the legend
        show (Boolean): Show or not the plot
    """

    for key in dico_callback:
        plt.plot(dico_callback[key], label=key)

    if title is not None:
        plt.title(title)

    if legend:
        plt.legend()
    
    if xlabel is not None:
        plt.xlabel(xlabel)
    
    if ylabel is not None:
        plt.ylabel(ylabel)

    if xlim is not None:
        plt.xlim(xlim)

    if ylim is not None:
        plt.ylim(ylim)

    if saving:
        plt.savefig(save_path)

    if show:
        plt.show()
