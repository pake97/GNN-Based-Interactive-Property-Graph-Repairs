import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# Set seaborn colorblind palette
colors = sns.color_palette("colorblind", 12)

colors.pop(8)



# Define markers for each line (10 distinct markers)
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'P', 'X', '*', 'H']

df = pd.read_csv('faers/exp3_quality_summary.csv')

x = df['budget'].unique().tolist()
x.append(0)
x.sort()

fig = plt.figure(figsize=(10,5))
ax = plt.subplot(111)


x = [0,80,240,480, 540, 720, 960, 1200, 1440, 1680, 1920, 2160, 2403]
# for i in range(11):
#     theta = i / 10
#     subset = df[df['theta'] == theta].sort_values(by='budget')
#     subset = subset[subset['budget'].isin(x)]
#     print(subset)
#     if(i < 2):
#         plt.plot(x, [0.8691399662731872]*len(x), linestyle='--', color=colors[i], marker=markers[i],  label=f'LARA-{theta:.1f}')    
#     else:
#         plt.plot(subset['budget'], subset['combined_f1'], linestyle='-' if i >= 2 else '--', color=colors[i], marker=markers[i], label=f'LARA-{theta:.1f}')
    
for i in range(11):
    theta = i / 10
    subset = df[df['theta'] == theta].sort_values(by='budget')
    subset = subset[subset['budget'].isin(x)]

    # draw the curve (no markers)
    y = [0.8691399662731872]*len(x) if i < 2 else subset['combined_f1']
    xvals = x if i < 2 else subset['budget']
    plt.plot(
        xvals, y,
        linestyle='--' if i < 2 else '-',
        color=colors[i],
        marker=markers[i],
        markerfacecolor=colors[i],
        markeredgecolor=colors[i],
        markeredgewidth=1.2,
        fillstyle='none',
        label=f'LARA-{theta:.1f}'
    )

    # place markers point-by-point with conditional fill
    # full if certified01 == True
    # half (left) if certified01 == False and certified05 == True
    # empty if both are False
    if i >= 2:
        for _, row in subset.iterrows():
            if row['certified01']:
                fillstyle = 'full'
                mfc = colors[i]
            elif row['certified05']:
                fillstyle = 'left'   # visually distinguishes the 0.05 case
                mfc = colors[i]
            else:
                fillstyle = 'none'
                mfc = 'none'

            plt.plot(
                [row['budget']], [row['combined_f1']],
                linestyle='None',
                marker=markers[i],
                color=colors[i],        # edge color
                markerfacecolor=mfc,
                markeredgecolor=colors[i],
                markeredgewidth=1.2,
                fillstyle=fillstyle,
                markersize=7
            )

plt.xticks(x)
plt.xticks(rotation=45)
plt.title("Comparison of F1 Score vs Budget for Different Theta Values for FAERS")
plt.xlabel("BUDGET")
plt.ylabel("F1 SCORE")
plt.legend(loc = "lower right", ncol=2)
plt.tight_layout()
# # Shrink current axis's height by 10% on the bottom
# box = ax.get_position()
# ax.set_position([box.x0, box.y0 + box.height * 0.1,
#                  box.width, box.height * 0.9])

# # Put a legend below current axis
# ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.26), fontsize=8,
#            ncol=6)

plt.show()



# Define markers for each line (10 distinct markers),
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'P', 'X', '*', 'H']

df = pd.read_csv('finbench/exp3_quality_summary.csv')

x = df['budget'].unique().tolist()
x.append(0)
x.sort()
print(x)
plt.figure(figsize=(10,5))
x=[0,270.0, 900.0, 1800.0,  2700.0,  3600.0,  4500.0, 5400.0, 6300.0, 7200.0, 8100.0, 9000.0]
#x = [0,90,450, 900,1220, 1260,1272, 1620, 1800, 2700, 3240, 3600,4091, 4500, 4770,4920 ,4950, 5400, 5490, 6300, 7200, 8100, 9000]
# for i in range(11):
#     theta = i / 10
#     subset = df[df['theta'] == theta].sort_values(by='budget')
#     subset = subset[subset['budget'].isin(x)]
#     if(i < 3):
#         plt.plot(x, [0.46893271231790706]*len(x), linestyle='--', color=colors[i], marker=markers[i],  label=f'LARA-{theta:.1f}')    
#     else:
#         plt.plot(subset['budget'], subset['combined_f1'], linestyle='-' if i >= 2 else '--', color=colors[i], marker=markers[i], label=f'LARA-{theta:.1f}')

for i in range(11):
    theta = i / 10
    subset = df[df['theta'] == theta].sort_values(by='budget')
    subset = subset[subset['budget'].isin(x)]

    # draw the curve (no markers)
    y = [0.46893271231790706]*len(x) if i < 3 else subset['combined_f1']
    xvals = x if i < 3 else subset['budget']
    plt.plot(
        xvals, y,
        linestyle='--' if i <3 else '-',
        color=colors[i],
        marker=markers[i],
        markerfacecolor=colors[i],
        markeredgecolor=colors[i],
        markeredgewidth=1.2,
        fillstyle='none',
        label=f'LARA-{theta:.1f}'
    )

    # place markers point-by-point with conditional fill
    # full if certified01 == True
    # half (left) if certified01 == False and certified05 == True
    # empty if both are False
    if i >= 3:
        for _, row in subset.iterrows():
            if row['certified01']:
                fillstyle = 'full'
                mfc = colors[i]
            elif row['certified05']:
                fillstyle = 'left'   # visually distinguishes the 0.05 case
                mfc = colors[i]
            else:
                fillstyle = 'none'
                mfc = 'none'

            plt.plot(
                [row['budget']], [row['combined_f1']],
                linestyle='None',
                marker=markers[i],
                color=colors[i],        # edge color
                markerfacecolor=mfc,
                markeredgecolor=colors[i],
                markeredgewidth=1.2,
                fillstyle=fillstyle,
                markersize=7
            )
plt.xticks(x)
plt.xticks(rotation=45)
plt.title("Comparison of F1 Score vs Budget for Different Theta Values for FinBench")
plt.xlabel("BUDGET")
plt.ylabel("F1 SCORE")
plt.legend(ncol=2)
plt.tight_layout()
plt.show()


# Define markers for each line (10 distinct markers)
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'P', 'X', '*', 'H']

df = pd.read_csv('icij/exp3_quality_summary.csv')

x = df['budget'].unique().tolist()
x.append(0)
x.sort()

print(x)

plt.figure(figsize=(10,5))


x = [0, 2108.0,  3532.0, 7064.0, 10569.0, 14128.0, 17660.0, 21192.0, 24724.0, 28256.0, 31788.0,35320.0, 38861.0]

# for i in range(11):
#     theta = i / 10
#     subset = df[df['theta'] == theta].sort_values(by='budget')
#     subset = subset[subset['budget'].isin(x)]
#     if(i < 5):
#         plt.plot(x, [0.5125704433751062]*len(x), linestyle='--', color=colors[i], marker=markers[i],  label=f'LARA-{theta:.1f}')    
#     else:
#         plt.plot(subset['budget'], subset['combined_f1'], linestyle='-' if i >= 2 else '--', color=colors[i], marker=markers[i], label=f'LARA-{theta:.1f}')
    
for i in range(11):
    theta = i / 10
    subset = df[df['theta'] == theta].sort_values(by='budget')
    subset = subset[subset['budget'].isin(x)]

    # draw the curve (no markers)
    y = [0.46893271231790706]*len(x) if i < 5 else subset['combined_f1']
    xvals = x if i < 5 else subset['budget']
    plt.plot(
        xvals, y,
        linestyle='--' if i < 5 else '-',
        color=colors[i],
        marker=markers[i],
        markerfacecolor=colors[i],
        markeredgecolor=colors[i],
        markeredgewidth=1.2,
        fillstyle='none',
        label=f'LARA-{theta:.1f}'
    )

    # place markers point-by-point with conditional fill
    # full if certified01 == True
    # half (left) if certified01 == False and certified05 == True
    # empty if both are False
    if i >= 5:
        for _, row in subset.iterrows():
            if row['certified01']:
                fillstyle = 'full'
                mfc = colors[i]
            elif row['certified05']:
                fillstyle = 'left'   # visually distinguishes the 0.05 case
                mfc = colors[i]
            else:
                fillstyle = 'none'
                mfc = 'none'

            plt.plot(
                [row['budget']], [row['combined_f1']],
                linestyle='None',
                marker=markers[i],
                color=colors[i],        # edge color
                markerfacecolor=mfc,
                markeredgecolor=colors[i],
                markeredgewidth=1.2,
                fillstyle=fillstyle,
                markersize=7
            )
plt.xticks(x)
plt.xticks(rotation=45)
plt.title("Comparison of F1 Score vs Budget for Different Theta Values for ICIJ")
plt.xlabel("BUDGET")
plt.ylabel("F1 SCORE")
plt.legend(ncol=2)
plt.tight_layout()
plt.show()


markers = ['o', 's', 'D', '^', 'v', '<', '>', 'P', 'X', '*', 'H']

df = pd.read_csv('snb/exp3_quality_summary.csv')

x = df['budget'].unique().tolist()
x.append(0)
x.sort()

plt.figure(figsize=(10,5))

x = [0,   2620.0,  6549.0, 13098.0, 19647.0, 26196.0, 32745.0, 39294.0, 45843.0, 52392.0, 58941.0,65493.0, 72044.0]
for i in range(11):
    theta = i / 10
    subset = df[df['theta'] == theta].sort_values(by='budget')
    subset = subset[subset['budget'].isin(x)]

    # draw the curve (no markers)
    y = [0.46893271231790706]*len(x) if i < 2 else subset['combined_f1']
    xvals = x if i < 2 else subset['budget']
    plt.plot(
        xvals, y,
        linestyle='--' if i < 2 else '-',
        color=colors[i],
        marker=markers[i],
        markerfacecolor=colors[i],
        markeredgecolor=colors[i],
        markeredgewidth=1.2,
        fillstyle='none',
        label=f'LARA-{theta:.1f}'
    )

    # place markers point-by-point with conditional fill
    # full if certified01 == True
    # half (left) if certified01 == False and certified05 == True
    # empty if both are False
    if i >= 2:
        for _, row in subset.iterrows():
            if row['certified01']:
                fillstyle = 'full'
                mfc = colors[i]
            elif row['certified05']:
                fillstyle = 'left'   # visually distinguishes the 0.05 case
                mfc = colors[i]
            else:
                fillstyle = 'none'
                mfc = 'none'

            plt.plot(
                [row['budget']], [row['combined_f1']],
                linestyle='None',
                marker=markers[i],
                color=colors[i],        # edge color
                markerfacecolor=mfc,
                markeredgecolor=colors[i],
                markeredgewidth=1.2,
                fillstyle=fillstyle,
                markersize=7
            )
plt.xticks(x)
plt.xticks(rotation=45)
plt.title("Comparison of F1 Score vs Budget for Different Theta Values for SNB")
plt.xlabel("BUDGET")
plt.ylabel("F1 SCORE")
plt.legend(ncol=2)
plt.tight_layout()
plt.show()


