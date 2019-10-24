import matplotlib.pyplot as plt
import numpy as np
import plt_templates as template
from palette import *

import palette as p
print(' our defined palette' )
print(dir(p) ) #hier sieht man quasi jede definierte farbe
print( '##############################')
del p

lines_dashed = template.linestyles( linestyle='--', marker='x', mfc='yellow')
default_lines = template.linestyles()

x = np.arange(20)

fig, ax = template.default_figure( 1,2, figsize=(6,6)) 
ax[0].plot( x, x, **lines_dashed[0], label='x')
ax[1].plot( x, x**2, **default_lines[1], label='x^2')
ax[0] = template.add_legend( ax[0], 'top_left')
ax[1] = template.add_legend( ax[1], 'top_left')
plt.show()
