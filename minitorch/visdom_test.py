import visdom
import matplotlib.pyplot as plt
vis = visdom.Visdom()

# Loss goes down!
plt.plot([2.0, 1.0, 0.0], c="blue")
plt.title("Model Loss")

# Send to visdom
vis.matplot(plt, win="loss")
