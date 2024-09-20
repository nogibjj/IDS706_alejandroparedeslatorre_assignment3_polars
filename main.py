from lib.stats_aplt import Df_Stats

if __name__ == "__main__":
    houses_url = (
        "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    )
    houses = Df_Stats(houses_url)

    columns = ["crim", "zn", "indus", "chas", "rm", "age"]
    for col in columns:
        # print(houses.df)
        print(f"Mean of variable {col}: {houses.variable_mean(col)}")
        print(f"Median of variable {col}: {houses.variable_median(col)}")
        print(f"Standard Deviation of variable {col}: {houses.variable_std(col)}")

    print(houses.stats_summary(columns))
    graphs = []
    for c in columns:
        houses.plot_hist_var(c, False, "./assets/" + c + ".png")
        graphs.append(f"./assets/{c}.png")

    for c in columns:
        for c2 in columns:
            if c != c2:
                houses.plot_scatter_two_vars(
                    c, c2, False, "./assets/" + c + "_" + c2 + ".png"
                )
                graphs.append(f"./assets/{c}_{c2}.png")

    houses.generate_report(
        "Houses summary", "Houses_Report", graphs, columns, report_type="pdf"
    )
