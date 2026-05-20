import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('paradise/exp3_quality_summary.csv', header=0)

colors = sns.color_palette("colorblind", 12)

budget_values = [750.0,1134.0,11344.0]

#budget_values = [750.0,11344.0]
thetas = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

fig, axes = plt.subplots(
    nrows=2,
    ncols=3,
    figsize=(16, 8),
    sharex=False
)

for col, budget_value in enumerate(budget_values):

    filtered = df[
        df["budget"].eq(budget_value) &
        df["theta"].isin(thetas)
    ].copy()

    # =========================================================
    # Row 1: F1 score
    # =========================================================
    f1_plot_df = (
        filtered
        .set_index("theta")[[
            "combined_f1",
            "combined_f1_1",
            "combined_f1_2",
            "combined_f1_3"
        ]]
        .rename(columns={
            "combined_f1": "all",
            "combined_f1_1": "DC",
            "combined_f1_2": "FD",
            "combined_f1_3": "Key"
        })
        .reset_index()
        .melt(
            id_vars="theta",
            var_name="cost_type",
            value_name="f1_score"
        )
    )

    sns.barplot(
        data=f1_plot_df,
        x="theta",
        y="f1_score",
        hue="cost_type",
        ax=axes[0, col],
        palette=colors
    )

    axes[0, col].set_title(f"Budget = {budget_value:g}")
    axes[0, col].set_xlabel("Theta")
    axes[0, col].set_ylabel("F1 Score" if col == 0 else "")
    
    axes[0, col].legend(
    loc='lower center',
    ncol=2
    )
    
    # =========================================================
    # Row 2: Average cost per repair
    # =========================================================
    filtered["all"] = filtered["total_cost"] / filtered["num_repairs"]
    filtered["DC"] = filtered["1_total_cost"] / filtered["1_num_repairs"]
    filtered["FD"] = filtered["2_total_cost"] / filtered["2_num_repairs"]
    filtered["Key"] = filtered["3_total_cost"] / filtered["3_num_repairs"]

    cost_plot_df = (
        filtered[["theta", "all", "DC", "FD", "Key"]]
        .melt(
            id_vars="theta",
            var_name="cost_type",
            value_name="avg_cost_per_repair"
        )
    )

    
    
    sns.barplot(
        data=cost_plot_df,
        x="theta",
        y="avg_cost_per_repair",
        hue="cost_type",
        ax=axes[1, col],
        palette=colors
    )

    axes[1, col].set_xlabel("Theta")
    axes[1, col].set_ylabel(
        "Average Cost per Repair" if col == 0 else ""
    )

    # # Remove subplot legends
    # axes[0, col].get_legend().remove()
    # axes[1, col].get_legend().remove()
    axes[1, col].legend(
    loc='lower center',
        ncol=2
    )

# Shared legend
handles, labels = axes[0, 0].get_legend_handles_labels()

# fig.legend(
#     handles,
#     labels,
#     title="",
#     loc="upper center",
#     ncol=4,
#     bbox_to_anchor=(0.5, 1.03)
# )

fig.suptitle(
    "ICIJ F1 Score and Average Cost per Repair by Theta and Budget",
    fontsize=16,
    y=1.02
)

# Tight layout BEFORE savefig
plt.tight_layout()

plt.savefig(
    "paradise_quality_summary_grid.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()