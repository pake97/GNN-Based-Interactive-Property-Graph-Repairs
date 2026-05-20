import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set seaborn colorblind palette
colors = sns.color_palette("colorblind", 12)


#colors.pop(8)



# Define markers for each line (10 distinct markers)
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'P', 'X', '*', 'H']

df = pd.read_csv('faers/exp3_quality_summary.csv', header=0)

gnn_f1 = {}

for theta in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
     
    with open(f"faers/grdgs/{str(theta)}_f1.txt", "r") as f:
        
        lines = f.readlines()
        
        f1 = float(lines[3].strip().split(": ")[1])
        f1_1 = float(lines[0].strip().split(": ")[1])
        f1_2 = float(lines[1].strip().split(": ")[1])
        f1_3 = float(lines[2].strip().split(": ")[1])
        
        gnn_f1[theta] = (f1, f1_1, f1_2, f1_3)


x = df['budget'].unique().tolist()
x.append(0)
x.sort()

fig = plt.figure(figsize=(10,5))
ax = plt.subplot(111)


df.sort_values(by=['theta', 'budget'], inplace=True)


    
for i in range(11):
    theta = i / 10
    print(theta)
    # draw the curve (no markers)
    
    if theta in df['theta'].unique():
        y = [None] + df[df['theta'] == theta]['combined_f1'].tolist()
        if(len(y) != len(x)):
            while len(y) < len(x):
                y = [None] + y
        
        plt.plot(
            x, y,
            linestyle='-',
            color=colors[i],
            marker=markers[i],
            markerfacecolor=colors[i],
            markeredgecolor=colors[i],
            markeredgewidth=1.2,
            fillstyle='none',
            label=f'LARA-{theta:.1f}'
        )
    else:
        
        y= [gnn_f1[theta][0]] * len(x)
        print(len(x), len(y))
        plt.plot(
            x, y,
            linestyle='--',
            color=colors[i],
            marker=markers[i],
            markerfacecolor=colors[i],
            markeredgecolor=colors[i],
            markeredgewidth=1.2,
            fillstyle='none',
            label=f'LARA-{theta:.1f}'
        )
    # for _, row in subset.iterrows():
    #     if row['certified01']:
    #         fillstyle = 'full'
    #         mfc = colors[i]
    #     elif row['certified05']:
    #         fillstyle = 'left'   # visually distinguishes the 0.05 case
    #         mfc = colors[i]
    #     else:
    #         fillstyle = 'none'
    #         mfc = 'none'

    #     plt.plot(
    #         [row['budget']], [row['combined_f1']],
    #         linestyle='None',
    #         marker=markers[i],
    #         color=colors[i],        # edge color
    #         markerfacecolor=mfc,
    #         markeredgecolor=colors[i],
    #         markeredgewidth=1.2,
    #         fillstyle=fillstyle,
    #         markersize=7
    #     )

plt.xticks(x)
plt.xticks(rotation=45)
plt.title("Comparison of F1 Score vs Budget for Different Theta Values for FAERS")
plt.xlabel("BUDGET")
plt.ylabel("F1 SCORE")
plt.legend(loc = "lower right", ncol=2)
plt.tight_layout()

plt.show()





# # Define markers for each line (10 distinct markers)
# markers = ['o', 's', 'D', '^', 'v', '<', '>', 'P', 'X', '*', 'H']

# df = pd.read_csv('faers/exp3_quality_summary.csv', header=0)

# gnn_f1 = {}

# for theta in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
     
#     with open(f"faers/grdgs/{str(theta)}_f1.txt", "r") as f:
        
#         lines = f.readlines()
        
#         f1 = float(lines[3].strip().split(": ")[1])
#         f1_1 = float(lines[0].strip().split(": ")[1])
#         f1_2 = float(lines[1].strip().split(": ")[1])
#         f1_3 = float(lines[2].strip().split(": ")[1])
        
#         gnn_f1[theta] = (f1, f1_1, f1_2, f1_3)


# x = df['budget'].unique().tolist()
# x.append(0)
# x.sort()

# fig = plt.figure(figsize=(10,5))
# ax = plt.subplot(111)


# df.sort_values(by=['theta', 'budget'], inplace=True)


    
# for i in range(11):
#     theta = i / 10
    
#     # draw the curve (no markers)
    
#     if theta in df['theta'].unique():
#         #y = [None] + df[df['theta'] == theta]['combined_f1_1'].tolist()
#         y = [None] + (df[df['theta'] == theta]['combined_f1_1'] /df[df['theta'] == theta]['1_num_repairs']).tolist()
#         if(len(y) != len(x)):
#             while len(y) < len(x):
#                 y = [None] + y
        
#         plt.plot(
#             x, y,
#             linestyle='-',
#             color=colors[i],
#             marker=markers[i],
#             markerfacecolor=colors[i],
#             markeredgecolor=colors[i],
#             markeredgewidth=1.2,
#             fillstyle='none',
#             label=f'LARA-{theta:.1f}'
#         )
#     else:
#         y= [gnn_f1[theta][1]] * len(x)
#         plt.plot(
#             x, y,
#             linestyle='--',
#             color=colors[i],
#             marker=markers[i],
#             markerfacecolor=colors[i],
#             markeredgecolor=colors[i],
#             markeredgewidth=1.2,
#             fillstyle='none',
#             label=f'LARA-{theta:.1f}'
#         )
#     # for _, row in subset.iterrows():
#     #     if row['certified01']:
#     #         fillstyle = 'full'
#     #         mfc = colors[i]
#     #     elif row['certified05']:
#     #         fillstyle = 'left'   # visually distinguishes the 0.05 case
#     #         mfc = colors[i]
#     #     else:
#     #         fillstyle = 'none'
#     #         mfc = 'none'

#     #     plt.plot(
#     #         [row['budget']], [row['combined_f1']],
#     #         linestyle='None',
#     #         marker=markers[i],
#     #         color=colors[i],        # edge color
#     #         markerfacecolor=mfc,
#     #         markeredgecolor=colors[i],
#     #         markeredgewidth=1.2,
#     #         fillstyle=fillstyle,
#     #         markersize=7
#     #     )

# plt.xticks(x)
# plt.xticks(rotation=45)
# plt.title("Comparison of F1 Score vs Budget for Different Theta Values for FAERS")
# plt.xlabel("BUDGET")
# plt.ylabel("F1 SCORE")
# plt.legend(loc = "lower right", ncol=2)
# plt.tight_layout()

# plt.show()





# # Define markers for each line (10 distinct markers)
# markers = ['o', 's', 'D', '^', 'v', '<', '>', 'P', 'X', '*', 'H']

# df = pd.read_csv('faers/exp3_quality_summary.csv', header=0)

# gnn_f1 = {}

# for theta in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
     
#     with open(f"faers/grdgs/{str(theta)}_f1.txt", "r") as f:
        
#         lines = f.readlines()
        
#         f1 = float(lines[3].strip().split(": ")[1])
#         f1_1 = float(lines[0].strip().split(": ")[1])
#         f1_2 = float(lines[1].strip().split(": ")[1])
#         f1_3 = float(lines[2].strip().split(": ")[1])
        
#         gnn_f1[theta] = (f1, f1_1, f1_2, f1_3)


# x = df['budget'].unique().tolist()
# x.append(0)
# x.sort()

# fig = plt.figure(figsize=(10,5))
# ax = plt.subplot(111)


# df.sort_values(by=['theta', 'budget'], inplace=True)


    
# for i in range(11):
#     theta = i / 10
    
#     # draw the curve (no markers)
    
#     if theta in df['theta'].unique():
#         #y = [None] + df[df['theta'] == theta]['combined_f1_2'].tolist()
#         y = [None] + (df[df['theta'] == theta]['combined_f1_2'] /df[df['theta'] == theta]['2_num_repairs']).tolist()
#         if(len(y) != len(x)):
#             while len(y) < len(x):
#                 y = [None] + y
        
#         plt.plot(
#             x, y,
#             linestyle='-',
#             color=colors[i],
#             marker=markers[i],
#             markerfacecolor=colors[i],
#             markeredgecolor=colors[i],
#             markeredgewidth=1.2,
#             fillstyle='none',
#             label=f'LARA-{theta:.1f}'
#         )
#     else:
#         y= [gnn_f1[theta][2]] * len(x)
#         plt.plot(
#             x, y,
#             linestyle='--',
#             color=colors[i],
#             marker=markers[i],
#             markerfacecolor=colors[i],
#             markeredgecolor=colors[i],
#             markeredgewidth=1.2,
#             fillstyle='none',
#             label=f'LARA-{theta:.1f}'
#         )
#     # for _, row in subset.iterrows():
#     #     if row['certified01']:
#     #         fillstyle = 'full'
#     #         mfc = colors[i]
#     #     elif row['certified05']:
#     #         fillstyle = 'left'   # visually distinguishes the 0.05 case
#     #         mfc = colors[i]
#     #     else:
#     #         fillstyle = 'none'
#     #         mfc = 'none'

#     #     plt.plot(
#     #         [row['budget']], [row['combined_f1']],
#     #         linestyle='None',
#     #         marker=markers[i],
#     #         color=colors[i],        # edge color
#     #         markerfacecolor=mfc,
#     #         markeredgecolor=colors[i],
#     #         markeredgewidth=1.2,
#     #         fillstyle=fillstyle,
#     #         markersize=7
#     #     )

# plt.xticks(x)
# plt.xticks(rotation=45)
# plt.title("Comparison of F1 Score vs Budget for Different Theta Values for FAERS")
# plt.xlabel("BUDGET")
# plt.ylabel("F1 SCORE")
# plt.legend(loc = "lower right", ncol=2)
# plt.tight_layout()

# plt.show()





# # Define markers for each line (10 distinct markers)
# markers = ['o', 's', 'D', '^', 'v', '<', '>', 'P', 'X', '*', 'H']

# df = pd.read_csv('faers/exp3_quality_summary.csv', header=0)

# gnn_f1 = {}

# for theta in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
     
#     with open(f"faers/grdgs/{str(theta)}_f1.txt", "r") as f:
        
#         lines = f.readlines()
        
#         f1 = float(lines[3].strip().split(": ")[1])
#         f1_1 = float(lines[0].strip().split(": ")[1])
#         f1_2 = float(lines[1].strip().split(": ")[1])
#         f1_3 = float(lines[2].strip().split(": ")[1])
        
#         gnn_f1[theta] = (f1, f1_1, f1_2, f1_3)


# x = df['budget'].unique().tolist()
# x.append(0)
# x.sort()

# fig = plt.figure(figsize=(10,5))
# ax = plt.subplot(111)


# df.sort_values(by=['theta', 'budget'], inplace=True)


    
# for i in range(11):
#     theta = i / 10
    
#     # draw the curve (no markers)
    
#     if theta in df['theta'].unique():
#         #y = [None] + df[df['theta'] == theta]['combined_f1_3'].tolist()
#         y = [None] + (df[df['theta'] == theta]['combined_f1_3'] /df[df['theta'] == theta]['3_num_repairs']).tolist()
#         if(len(y) != len(x)):
#             while len(y) < len(x):
#                 y = [None] + y
#         plt.plot(
#             x, y,
#             linestyle='-',
#             color=colors[i],
#             marker=markers[i],
#             markerfacecolor=colors[i],
#             markeredgecolor=colors[i],
#             markeredgewidth=1.2,
#             fillstyle='none',
#             label=f'LARA-{theta:.1f}'
#         )
#     else:
#         y= [gnn_f1[theta][3]] * len(x)
#         plt.plot(
#             x, y,
#             linestyle='--',
#             color=colors[i],
#             marker=markers[i],
#             markerfacecolor=colors[i],
#             markeredgecolor=colors[i],
#             markeredgewidth=1.2,
#             fillstyle='none',
#             label=f'LARA-{theta:.1f}'
#         )
#     # for _, row in subset.iterrows():
#     #     if row['certified01']:
#     #         fillstyle = 'full'
#     #         mfc = colors[i]
#     #     elif row['certified05']:
#     #         fillstyle = 'left'   # visually distinguishes the 0.05 case
#     #         mfc = colors[i]
#     #     else:
#     #         fillstyle = 'none'
#     #         mfc = 'none'

#     #     plt.plot(
#     #         [row['budget']], [row['combined_f1']],
#     #         linestyle='None',
#     #         marker=markers[i],
#     #         color=colors[i],        # edge color
#     #         markerfacecolor=mfc,
#     #         markeredgecolor=colors[i],
#     #         markeredgewidth=1.2,
#     #         fillstyle=fillstyle,
#     #         markersize=7
#     #     )

# plt.xticks(x)
# plt.xticks(rotation=45)
# plt.title("Comparison of F1 Score vs Budget for Different Theta Values for FAERS")
# plt.xlabel("BUDGET")
# plt.ylabel("F1 SCORE")
# plt.legend(loc = "lower right", ncol=2)
# plt.tight_layout()

# plt.show()



# Define markers for each line (10 distinct markers)
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'P', 'X', '*', 'H']

df = pd.read_csv('paradise/exp3_quality_summary.csv', header=0)

gnn_f1 = {}

for theta in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
     
    with open(f"paradise/grdgs/{str(theta)}_f1.txt", "r") as f:
        
        lines = f.readlines()
        
        f1 = float(lines[3].strip().split(": ")[1])
        f1_1 = float(lines[0].strip().split(": ")[1])
        f1_2 = float(lines[1].strip().split(": ")[1])
        f1_3 = float(lines[2].strip().split(": ")[1])
        
        gnn_f1[theta] = (f1, f1_1, f1_2, f1_3)


x = df['budget'].unique().tolist()
x.append(0)
x.sort()

fig = plt.figure(figsize=(10,5))
ax = plt.subplot(111)


df.sort_values(by=['theta', 'budget'], inplace=True)


    
for i in range(11):
    theta = i / 10
    print(theta)
    # draw the curve (no markers)
    
    if theta in df['theta'].unique():
        y = [None] + df[df['theta'] == theta]['combined_f1'].tolist()
        if(len(y) != len(x)):
            while len(y) < len(x):
                y = [None] + y
        
        plt.plot(
            x, y,
            linestyle='-',
            color=colors[i],
            marker=markers[i],
            markerfacecolor=colors[i],
            markeredgecolor=colors[i],
            markeredgewidth=1.2,
            fillstyle='none',
            label=f'LARA-{theta:.1f}'
        )
    else:
        
        y= [gnn_f1[theta][0]] * len(x)
        print(len(x), len(y))
        plt.plot(
            x, y,
            linestyle='--',
            color=colors[i],
            marker=markers[i],
            markerfacecolor=colors[i],
            markeredgecolor=colors[i],
            markeredgewidth=1.2,
            fillstyle='none',
            label=f'LARA-{theta:.1f}'
        )
    # for _, row in subset.iterrows():
    #     if row['certified01']:
    #         fillstyle = 'full'
    #         mfc = colors[i]
    #     elif row['certified05']:
    #         fillstyle = 'left'   # visually distinguishes the 0.05 case
    #         mfc = colors[i]
    #     else:
    #         fillstyle = 'none'
    #         mfc = 'none'

    #     plt.plot(
    #         [row['budget']], [row['combined_f1']],
    #         linestyle='None',
    #         marker=markers[i],
    #         color=colors[i],        # edge color
    #         markerfacecolor=mfc,
    #         markeredgecolor=colors[i],
    #         markeredgewidth=1.2,
    #         fillstyle=fillstyle,
    #         markersize=7
    #     )

plt.xticks(x)
plt.xticks(rotation=45)
plt.title("Comparison of F1 Score vs Budget for Different Theta Values for FAERS")
plt.xlabel("BUDGET")
plt.ylabel("F1 SCORE")
plt.legend(loc = "lower right", ncol=2)
plt.tight_layout()

plt.show()
