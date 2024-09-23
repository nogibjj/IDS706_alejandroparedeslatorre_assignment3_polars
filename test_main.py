from lib.stats_aplt import Df_Stats
import os

houses_url = (
    "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
)
houses = Df_Stats(houses_url)


def test_mean():
    """Testing mean"""
    print(houses.variable_mean("crim"))
    assert houses.variable_mean("crim") == 3.613523557312253
    print("Success")


def test_median():
    """Testing median"""

    assert houses.variable_median("crim") == 0.25651
    print("Success")


def test_std():
    """Testing standard deviation"""

    assert houses.variable_std("crim") == 8.60154510533249
    print("Success")


def test_plot_hist_var():
    """Testing histogram plot creation"""

    filename = "./assets/test_histogram.png"
    houses.plot_hist_var("crim", False, filename)

    assert os.path.exists(filename), "Histogram plot was not created."
    print("Histogram plot creation test passed")

    # Clean up the file after the test
    os.remove(filename)


def test_plot_scatter_two_vars():
    """Testing scatter plot creation"""

    filename = "./assets/test_scatter.png"
    houses.plot_scatter_two_vars("crim", "indus", False, filename)

    assert os.path.exists(filename), "Scatter plot was not created."
    print("Scatter plot creation test passed")

    # Clean up the file after the test
    os.remove(filename)


if __name__ == "__main__":
    test_mean()
    test_median()
    test_std()
    # Test plot functions
    test_plot_hist_var()
    test_plot_scatter_two_vars()
