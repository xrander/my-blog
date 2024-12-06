# Import library -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## Set style and themes for plots ---------------------
sns.set_style("whitegrid")
plt.style.use("tableau-colorblind10")


# Import data --------------------------------
penguins = pd.read_csv("penguins.csv")
penguins.head()

# Find and Remove missing data
penguins.isna().sum()


## Rows with one or more missing data
penguins.loc[penguins.isna().any(axis=1)]
print(f" The total number of missing observation is: {len(penguins.loc[penguins.isna().any(axis=1)])}")

penguins_cleaned = penguins.dropna()

penguins_cleaned.info()
penguins_cleaned.isna().sum() # rows with NAs removed


# Exploratory Data Analysis ------------------------------------

## Total number of species

print(f"The number of species on the island is: {len(penguins_cleaned.species.unique())}")
penguins_cleaned.species.unique()

## Total number of islands
len(penguins_cleaned.island.unique())
penguins_cleaned.island.unique()

## Total number of penguins according to species in the clean_data ---

penguins_cleaned.groupby("species").agg(
    species_count = pd.NamedAgg(column="species", aggfunc="count")
).sort_values("species_count", ascending=False)

### alternatively
penguins_cleaned.species.value_counts()

## Number of penguins in the island

penguins_cleaned.groupby("island").agg(
    island_count = pd.NamedAgg(column="id", aggfunc="count")
)

penguins_cleaned.groupby("sex").agg(
    sex_count = pd.NamedAgg(column="id", aggfunc="count")
)

## Average Body Mass for Each Species
penguins_cleaned.groupby("species").agg(
    mean_weight = pd.NamedAgg(column="body_mass_g", aggfunc="mean")
)


## Investigate the distribution of flipper_length_mm
adelie_species = penguins_cleaned[penguins_cleaned.species == "Adelie"]
gentoo_species = penguins_cleaned[penguins_cleaned.species == "Gentoo"]
chinstrap_species = penguins_cleaned[penguins_cleaned.species == "Chinstrap"]



plt.figure()
plt.hist(adelie_species.flipper_length_mm, label="Adelie")
plt.hist(gentoo_species.flipper_length_mm, label="Gentoo")
plt.hist(chinstrap_species.flipper_length_mm, label="Chinstrap")
plt.title(
    "Distribution of Flipper Length for the Three Penguin Species", 
    size=15, loc="left", fontweight="bold"
)
plt.xlabel("Flipper Length ($mm$)", size=10, fontweight="bold")
plt.ylabel("Count", size=10, weight="bold")
plt.legend()

species_group_fl = penguins_cleaned.groupby("species")["flipper_length_mm"]

fig, ax = plt.subplots(figsize=(8,6))

boxplot = ax.boxplot(
    x=[species.values for name, species in species_group_fl],
    vert=False,
    tick_labels=species_group_fl.groups.keys(),
    patch_artist=True,
    medianprops={'color': 'black'}
) 

# Define colors for each group
colors = ['orange', 'purple', "red"]

# Assign colors to each box in the boxplot
for box, color in zip(boxplot['boxes'], colors):
    box.set_facecolor(color)

plt.title(
    label="Distribution of Flipper Length Across Species",
     size = 23, weight="bold"
)
ax.set_ylabel("Species", size=13, loc="top", weight="bold")
ax.set_xlabel("Flipper Length ($mm$)", size=13, loc="right", weight="bold")

plt.clf() # clear plot

# Alternatively 
sns.boxplot(
    x="flipper_length_mm",
    y="species",
    data=penguins_cleaned,
    hue="species"
)
plt.xlabel("Flipper length ($mm$)", weight="bold", size=13)
plt.ylabel("Species", weight="bold", size=13)
plt.title("Distribution of Flipperlength According to Species", size=20, weight="bold")


## Investigation of relationship
plt.figure()
sns.scatterplot(
    x="bill_length_mm",
    y="bill_depth_mm",
    data=penguins_cleaned,
    hue="species",  
)
rel_plt = sns.lmplot(
    x="bill_length_mm",
    y="bill_depth_mm",
    data=penguins_cleaned,
    hue="species",
    markers=["+", "o", "p"]
)
plt.tight_layout()
sns.move_legend(
    rel_plt,
    "upper right",
    frameon=True
)
plt.title("Relationship between Bill length($mm$) and Bill depth($mm$)", size=22, weight="bold")
plt.xlabel("Bill length ($mm$)", size=12)
plt.ylabel("Bill depth ($mm$)", size=12)

## How the average mass compares across the sexes
penguins_cleaned.groupby("sex")["body_mass_g"].mean().round(2)
penguins_cleaned.groupby("sex").agg(
    mean_weight = pd.NamedAgg(column="body_mass_g", aggfunc="mean")
)

plt_dt = penguins_cleaned.groupby(["sex", "species"])["body_mass_g"].agg("mean")

plt_dt = plt_dt.reset_index()
plt.figure()

sns.barplot(
    x="sex", y="body_mass_g",
    hue="species", data=plt_dt
)
plt.xlabel("Sex", weight="bold", loc="right", size=13)
plt.ylabel("Body Mass ($g$)", weight="bold", loc="top", size=13)
plt.title("Average weight ($g$) of Penguins Specices")
plt.legend(title="Species", title_fontproperties={"weight": "bold"})

# Proportion of penguin species found on each island?

penguins_cleaned.value_counts(
    subset=["island","species"],
     normalize=True, sort=False
     ).round(2).reset_index()

# How does flipper length vary across the different species and islands?

plt.figure()
sns.boxplot(
    x="island",
    y="flipper_length_mm",
    hue="species",
    data=penguins_cleaned
)
plt.xlabel("Island", loc="right", size=9, weight=900, style="italic")
plt.ylabel("Flipper Length ($mm$)",size=9, loc="top", style="italic", weight=900)
plt.title(
    "Distribution of Flipper Length ($mm$), Penguins Species Across Different Islands",
    weight="heavy", size=14
)
# Find max and min body_mass
## max
penguins_cleaned[penguins_cleaned.body_mass_g == penguins_cleaned.body_mass_g.max()]
penguins_cleaned.body_mass_g.max()

## min
penguins_cleaned[penguins_cleaned.body_mass_g == penguins_cleaned.body_mass_g.min()]
penguins_cleaned.body_mass_g.min()
