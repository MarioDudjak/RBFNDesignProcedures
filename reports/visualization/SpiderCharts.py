import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
from pprint import pprint

font = {'family': 'normal',
        'weight': 'normal',
        'size': 12}
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

## for Palatino and other serif fonts use:
rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)
datasetLabels = [r'$\mathcal{D}_1$', r'$\mathcal{D}_2$', r'$\mathcal{D}_3$', r'$\mathcal{D}_4$', r'$\mathcal{D}_5$',
                 r'$\mathcal{D}_6$', r'$\mathcal{D}_7$', r'$\mathcal{D}_8$', r'$\mathcal{D}_9$', r'$\mathcal{D}_{10}$',
                 r'$\mathcal{D}_{11}$', r'$\mathcal{D}_{12}$']

i = 0
cmap = 'flare_r'
rows = 1
cols = 1
colors = iter([plt.cm.Set2(i) for i in range(7)])
data = [[2, 1.47, 12.23, 0.83, 3.77, 0.90, 9.33, 6.00, 9.37, 7.13, 4.00, 0.97],
        [1.83, 1.47, 9.46, 1.07, 3.60, 1.70, 8.43, 3.47, 6.03, 4.67, 1.53, 1.87]]

data2 = [[0.87, 0.90, 12.17, 1.67, 3.77, 0.77, 6.60, 7.07, 10.97, 2.47, 0.47, 0.17],
         [5.83, 1.83, 18.90, 13.70, 5.17, 1.43, 17.17, 4.47, 16.00, 3.70, 5.93, 3.83],
         [5.33, 5.83, 15.40, 3.53, 7.40, 8.03, 16.03, 10.20, 8.30, 8.43, 5.20, 3.83],
         [7.10, 11.23, 18.87, 10.47, 6.47, 10.47, 8.93, 11.57, 19, 7.90, 11.43, 9.90]
         ]
i = 0
angles = [n / 12 * 2 * 3.14 for n in range(12)]
angles += angles[:1]

c = [next(colors)][0]
fig, axes = plt.subplots(nrows=rows, ncols=cols, constrained_layout=True, subplot_kw={'projection': 'polar'})
ax = axes
ax.set_theta_offset(3.14 / 6)
ax.set_theta_direction(-1)
# ax.set_xticks(angles[:-1], measures)
ax.tick_params(axis='x', rotation=1.5, labelsize=13, pad=0)
ax.tick_params(axis='y', labelsize=13)
ax.set_rlabel_position(18)
plt.setp(axes, xticks=angles[:-1], xticklabels=datasetLabels, yticks=[1, 4, 7, 10, 13, 16, 19])
# plt.yticks([20, 40, 60, 80], ["20", "40", "60", "80"], color="black", size=10)
ax.set_ylim(0, 13)
values = data[i]
values += values[:1]
im = ax.plot(angles, values, linewidth=1, linestyle='solid', color=c)
ax.fill(angles, values, color=c)
# ax.set_xlabel(key, fontsize=18)
# ax.title.set_text(f"{datasetsNames[i]}")
# ax.title.set_size(5)
# plt.show()
# plt.subplots_adjust(hspace=-0.9, wspace=1.6)
plt.tight_layout()
# plt.show()
plt.savefig(f'Ir_spider.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.close()

i += 1

c = [next(colors)][0]
fig, axes = plt.subplots(nrows=rows, ncols=cols, constrained_layout=True, subplot_kw={'projection': 'polar'})
ax = axes
ax.set_theta_offset(3.14 / 6)
ax.set_theta_direction(-1)
# ax.set_xticks(angles[:-1], measures)
ax.tick_params(axis='x', rotation=1.5, labelsize=15, pad=0)
ax.tick_params(axis='y', labelsize=15)
ax.set_rlabel_position(18)
plt.setp(axes, xticks=angles[:-1], xticklabels=datasetLabels, yticks=[1, 4, 7, 10, 13, 16, 19])
# plt.yticks([20, 40, 60, 80], ["20", "40", "60", "80"], color="black", size=10)
ax.set_ylim(0, 13)
values = data[i]
values += values[:1]
im = ax.plot(angles, values, linewidth=1, linestyle='solid', color=c)
ax.fill(angles, values, color=c)
# ax.set_xlabel(key, fontsize=18)
# ax.title.set_text(f"{datasetsNames[i]}")
# ax.title.set_size(5)
# plt.show()
# plt.subplots_adjust(hspace=-0.9, wspace=1.6)
plt.tight_layout()
# plt.show()
plt.savefig(f'Ie_spider.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.close()

i = 0
c = [next(colors)][0]
fig, axes = plt.subplots(nrows=rows, ncols=cols, constrained_layout=True, subplot_kw={'projection': 'polar'})
ax = axes
ax.set_theta_offset(3.14 / 6)
ax.set_theta_direction(-1)
# ax.set_xticks(angles[:-1], measures)
ax.tick_params(axis='x', rotation=1.5, labelsize=13, pad=0)
ax.tick_params(axis='y', labelsize=13)
ax.set_rlabel_position(18)
plt.setp(axes, xticks=angles[:-1], xticklabels=datasetLabels, yticks=[1, 4, 7, 10, 13, 16, 19])
# plt.yticks([20, 40, 60, 80], ["20", "40", "60", "80"], color="black", size=10)
ax.set_ylim(0, 19)
values = data2[i]
values += values[:1]
im = ax.plot(angles, values, linewidth=1, linestyle='solid', color=c)
ax.fill(angles, values, color=c)
# ax.set_xlabel(key, fontsize=18)
# ax.title.set_text(f"{datasetsNames[i]}")
# ax.title.set_size(5)
# plt.show()
# plt.subplots_adjust(hspace=-0.9, wspace=1.6)
plt.tight_layout()
# plt.show()
plt.savefig(f'Jpso_spider.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.close()

i += 1

c = [next(colors)][0]
fig, axes = plt.subplots(nrows=rows, ncols=cols, constrained_layout=True, subplot_kw={'projection': 'polar'})
ax = axes
ax.set_theta_offset(3.14 / 6)
ax.set_theta_direction(-1)
# ax.set_xticks(angles[:-1], measures)
ax.tick_params(axis='x', rotation=1.5, labelsize=13, pad=0)
ax.tick_params(axis='y', labelsize=13)
ax.set_rlabel_position(18)
plt.setp(axes, xticks=angles[:-1], xticklabels=datasetLabels, yticks=[1, 4, 7, 10, 13, 16, 19])
# plt.yticks([20, 40, 60, 80], ["20", "40", "60", "80"], color="black", size=10)
ax.set_ylim(0, 19)
values = data2[i]
values += values[:1]
im = ax.plot(angles, values, linewidth=1, linestyle='solid', color=c)
ax.fill(angles, values, color=c)
# ax.set_xlabel(key, fontsize=18)
# ax.title.set_text(f"{datasetsNames[i]}")
# ax.title.set_size(5)
# plt.show()
# plt.subplots_adjust(hspace=-0.9, wspace=1.6)
plt.tight_layout()
# plt.show()
plt.savefig(f'Jkmeans_spider.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.close()

i += 1

c = [next(colors)][0]
fig, axes = plt.subplots(nrows=rows, ncols=cols, constrained_layout=True, subplot_kw={'projection': 'polar'})
ax = axes
ax.set_theta_offset(3.14 / 6)
ax.set_theta_direction(-1)
# ax.set_xticks(angles[:-1], measures)
ax.tick_params(axis='x', rotation=1.5, labelsize=13, pad=0)
ax.tick_params(axis='y', labelsize=13)
ax.set_rlabel_position(18)
plt.setp(axes, xticks=angles[:-1], xticklabels=datasetLabels, yticks=[1, 4, 7, 10, 13, 16, 19])
# plt.yticks([20, 40, 60, 80], ["20", "40", "60", "80"], color="black", size=10)
ax.set_ylim(0, 19)
values = data2[i]
values += values[:1]
im = ax.plot(angles, values, linewidth=1, linestyle='solid', color=c)
ax.fill(angles, values, color=c)
# ax.set_xlabel(key, fontsize=18)
# ax.title.set_text(f"{datasetsNames[i]}")
# ax.title.set_size(5)
# plt.show()
# plt.subplots_adjust(hspace=-0.9, wspace=1.6)
plt.tight_layout()
# plt.show()
plt.savefig(f'Ade_spider.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.close()

i += 1

c = [next(colors)][0]
fig, axes = plt.subplots(nrows=rows, ncols=cols, constrained_layout=True, subplot_kw={'projection': 'polar'})
ax = axes
ax.set_theta_offset(3.14 / 6)
ax.set_theta_direction(-1)
# ax.set_xticks(angles[:-1], measures)
ax.tick_params(axis='x', rotation=1.5, labelsize=13, pad=0)
ax.tick_params(axis='y', labelsize=13)
ax.set_rlabel_position(18)
plt.setp(axes, xticks=angles[:-1], xticklabels=datasetLabels, yticks=[1, 4, 7, 10, 13, 16, 19])
# plt.yticks([20, 40, 60, 80], ["20", "40", "60", "80"], color="black", size=10)
ax.set_ylim(0, 19)
values = data2[i]
values += values[:1]
im = ax.plot(angles, values, linewidth=1, linestyle='solid', color=c)
ax.fill(angles, values, color=c)
# ax.set_xlabel(key, fontsize=18)
# ax.title.set_text(f"{datasetsNames[i]}")
# ax.title.set_size(5)
# plt.show()
# plt.subplots_adjust(hspace=-0.9, wspace=1.6)
plt.tight_layout()
# plt.show()
plt.savefig(f'Apso_spider.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.close()

i += 1
