import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# Set seaborn colorblind palette
colors = sns.color_palette("colorblind", 20)





# Define markers for each line (10 distinct markers)
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'P', 'X', '*', 'H']

df1 = pd.read_csv('faers/exp4_runtime_summary.csv')
df2 = pd.read_csv('finbench/exp4_runtime_summary.csv')
df3 = pd.read_csv('icij/exp4_runtime_summary.csv')
df4 = pd.read_csv('snb/exp4_runtime_summary.csv')


print(df1)

fig = plt.figure(figsize=(8, 4))
ax = plt.subplot(111)


x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

b1max = df1['budget'].max()
b2max = df2['budget'].max()
b3max = df3['budget'].max()
b4max = df4['budget'].max()


subset1 = df1[df1['budget'] == b1max].sort_values(by='theta')
plt.plot(subset1['theta'], subset1['total_time'], linestyle='-', label=f'FAERS', color=colors[0], marker=markers[0])
subset2 = df2[df2['budget'] == b2max].sort_values(by='theta')
plt.plot(subset2['theta'], subset2['total_time'], linestyle='-', label=f'FinBench', color=colors[1], marker=markers[1])
subset3 = df3[df3['budget'] == b3max].sort_values(by='theta')
plt.plot(subset3['theta'], subset3['total_time'], linestyle='-', label=f'ICIJ', color=colors[2], marker=markers[2])
subset4 = df4[df4['budget'] == b4max].sort_values(by='theta')
plt.plot(subset4['theta'], subset4['total_time'], linestyle='-', label=f'SNB', color=colors[3], marker=markers[3])






# for i in range(11):
#     theta = i / 10
#     subset = df[df['theta'] == theta].sort_values(by='budget')
#     subset = subset[subset['budget'].isin(x)]
#     print(subset)
#     if(i < 2):
#         plt.plot(x, [0.8691399662731872]*len(x), linestyle='--', color=colors[i], marker=markers[i],  label=f'LARA-{theta:.1f}')    
#     else:
#         plt.plot(subset['budget'], subset['combined_f1'], linestyle='-' if i >= 2 else '--', color=colors[i], marker=markers[i], label=f'LARA-{theta:.1f}')

    

plt.xticks(x)
plt.title("Runtime vs Theta (max budget)")
plt.xlabel("Theta")
plt.ylabel("Time (seconds)")
plt.tight_layout()
plt.legend()





# # Shrink current axis's height by 10% on the bottom
# # box = ax.get_position()
# # ax.set_position([box.x0, box.y0 + box.height * 0.1,
# #                  box.width, box.height * 0.9])

# # # Put a legend below current axis
# # ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.26), fontsize=8,
# #            ncol=6)

plt.show()



# # Define markers for each line (10 distinct markers),
# markers = ['o', 's', 'D', '^', 'v', '<', '>', 'P', 'X', '*', 'H']

# df = pd.read_csv('finbench/exp4_runtime_summary.csv')

# x = df['budget'].unique().tolist()
# x.append(0)
# x.sort()
# print(x)
# plt.figure(figsize=(10, 6))
# x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# unique_badgets = df['budget'].unique().tolist()
# unique_badgets.sort()
# print(unique_badgets)
# for idx,b in enumerate(unique_badgets):
#     print(idx)
#     print(colors[idx])
    
#     subset = df[df['budget'] == b].sort_values(by='theta')
#     plt.plot(subset['theta'], subset['total_time'], linestyle='-', label=f'Budget {b}', color=colors[idx])


# plt.xticks(x)

# plt.title("Comparison of F1 Score vs Budget for Different Theta Values for FinBench")
# plt.xlabel("BUDGET")
# plt.ylabel("F1 SCORE")
# plt.legend()
# plt.show()


# # Define markers for each line (10 distinct markers)
# markers = ['o', 's', 'D', '^', 'v', '<', '>', 'P', 'X', '*', 'H']

# df = pd.read_csv('icij/exp4_runtime_summary.csv')

# x = df['budget'].unique().tolist()
# x.append(0)
# x.sort()

# print(x)

# plt.figure(figsize=(10, 6.3))


# x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# # for i in range(11):
# #     theta = i / 10
# #     subset = df[df['theta'] == theta].sort_values(by='budget')
# #     subset = subset[subset['budget'].isin(x)]
# #     if(i < 5):
# #         plt.plot(x, [0.5125704433751062]*len(x), linestyle='--', color=colors[i], marker=markers[i],  label=f'LARA-{theta:.1f}')    
# #     else:
# #         plt.plot(subset['budget'], subset['combined_f1'], linestyle='-' if i >= 2 else '--', color=colors[i], marker=markers[i], label=f'LARA-{theta:.1f}')
# unique_badgets = df['budget'].unique().tolist()
# unique_badgets.sort()
# print(unique_badgets)
# for idx,b in enumerate(unique_badgets):
#     print(idx)
#     print(colors[idx])
    
#     subset = df[df['budget'] == b].sort_values(by='theta')
#     plt.plot(subset['theta'], subset['total_time'], linestyle='-', label=f'Budget {b}', color=colors[idx])

# plt.xticks(x)

# plt.title("Comparison of F1 Score vs Budget for Different Theta Values for ICIJ")
# plt.xlabel("BUDGET")
# plt.ylabel("F1 SCORE")
# plt.legend()
# plt.show()


# markers = ['o', 's', 'D', '^', 'v', '<', '>', 'P', 'X', '*', 'H']

# df = pd.read_csv('snb/exp4_runtime_summary.csv')

# x = df['budget'].unique().tolist()
# x.append(0)
# x.sort()

# plt.figure(figsize=(10, 6.1))

# x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# unique_badgets = df['budget'].unique().tolist()
# unique_badgets.sort()
# print(unique_badgets)
# for idx,b in enumerate(unique_badgets):
#     print(idx)
#     print(colors[idx])
    
#     subset = df[df['budget'] == b].sort_values(by='theta')
#     plt.plot(subset['theta'], subset['total_time'], linestyle='-', label=f'Budget {b}', color=colors[idx])

# plt.xticks(x)

# plt.title("Comparison of F1 Score vs Budget for Different Theta Values for SNB")
# plt.xlabel("BUDGET")
# plt.ylabel("F1 SCORE")
# plt.legend()
# plt.show()




# avg_total_time1 = df1.groupby(['dataset', 'theta'])['total_time'].mean().reset_index()
# avg_total_time2 = df2.groupby(['dataset', 'theta'])['total_time'].mean().reset_index()
# avg_total_time3 = df3.groupby(['dataset', 'theta'])['total_time'].mean().reset_index()
# avg_total_time4 = df4.groupby(['dataset', 'theta'])['total_time'].mean().reset_index()



# plt.plot(avg_total_time1['theta'], avg_total_time1['total_time'], linestyle='-', label=f'FAERS', color=colors[0], marker=markers[0])
# plt.plot(avg_total_time2['theta'], avg_total_time2['total_time'], linestyle='-', label=f'FinBench', color=colors[1], marker=markers[1])
# plt.plot(avg_total_time3['theta'], avg_total_time3['total_time'], linestyle='-', label=f'ICIJ', color=colors[2], marker=markers[2])
# plt.plot(avg_total_time4['theta'], avg_total_time4['total_time'], linestyle='-', label=f'SNB', color=colors[3], marker=markers[3])




# plt.xticks(x)
# plt.title("Runtime vs Theta (max budget)")
# plt.xlabel("Theta")
# plt.ylabel("Time (seconds)")
# plt.tight_layout()
# plt.legend()

# plt.show()