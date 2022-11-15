import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

color = {"Finch": "tab:red",
"Finch (Gallop)": "tab:orange",
"Finch (Lead)": "tab:brown",
"Finch (Follow)": "tab:grey",
"Finch (VBL)": "tab:olive",
"Finch (RLE)": "tab:grey",
"TACO (RLE)": "tab:purple",
"TACO": "tab:purple",
"OpenCV": "tab:blue"}

font = {'size'   : 14}

plt.rc('font', **font)

frame = pd.read_csv('ReadySpMSpVCount.csv')
viz = frame.plot(kind="box", figsize=(5,3.5), rot=20, ylabel="Speedup Over TACO", yticks=[1, 2, 3, 4, 5, 6, 7], ylim=[0, 7])
viz.axes.axhline(1, color="grey")
viz.get_figure().savefig("images/spmspv10count.png", bbox_inches="tight")

frame = pd.read_csv('ReadySpMSpVDense.csv')
viz = frame.plot(kind="box", figsize=(5,3.5), rot=20, ylabel="Speedup Over TACO", yticks=[1, 3, 5, 7, 9, 11])
viz.axes.axhline(1, color="grey")
viz.get_figure().savefig("images/spmspv10dense.png", bbox_inches="tight")

frame = pd.read_csv('ReadyTriangle.csv')
viz = frame.plot(kind="box", figsize=(4,3.5), yticks=[1, 2, 3, 4])
viz.axes.axhline(1, color="grey")
viz.get_figure().savefig("images/triangle.png", bbox_inches="tight")

frame = pd.read_csv('ReadyAllPairs.csv')
viz = frame.plot(kind="bar", figsize=(10,3.5), x="Dataset", ylabel = "Speedup Over OpenCV", rot=0, yticks = [y*0.2 for y in range(6)], color=color)
viz.axes.get_xaxis().get_label().set_visible(False)
viz.axes.axhline(1, color="grey")
viz.legend(loc="upper left")
viz.get_figure().savefig("images/allpairs.png", bbox_inches="tight")

frame = pd.read_csv('ReadyAlpha.csv')
viz = frame.plot(kind="bar", figsize=(5,3.5), x="Dataset", ylabel = "Speedup Over OpenCV", rot=0, color=color)#, yticks = [y*0.1 for y in range(11)])
viz.axes.get_xaxis().get_label().set_visible(False)
viz.legend(loc="upper left")
viz.axes.axhline(1, color="grey")
viz.get_figure().savefig("images/alpha.png", bbox_inches="tight")

frame = pd.read_csv('ReadyConv.csv')
viz = frame.plot(kind="line", figsize=(5,3.5), x="Sparsity (% Nonzero)", loglog=True, ylabel="Runtime (s)", color=color, style=["o--", "x-"])
viz.invert_xaxis()
viz.get_figure().savefig("images/conv.png", bbox_inches="tight")