# Alejandro Paredes La Torre
import polars as pl
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
import numpy as np
import statistics as stat
from fpdf import FPDF
import os

class Df_Stats:
    """ Returns principal stats and summary statistics and prints a report """
    def __init__(self, url):
        self.df = pl.read_csv(url)
    
    def variable_mean(self, variable):
        return self.df[variable].mean()

    def variable_median(self, variable):
        return stat.median(self.df[variable].to_list())

    def variable_std(self, variable):
        return self.df[variable].std()

    def variable_percentile(self, variable, percentile):
        return np.percentile(self.df[variable].to_numpy(), percentile)

    def stats_summary(self, variables=None):
        if variables:
            return self.df.select(variables).describe()
        return self.df.describe()

    def plot_hist_var(self, var, jupyter_render, filename="plot_var.png"):
        plt.hist(self.df[var].to_numpy(), bins=30, color='skyblue', edgecolor='black')  # Create histogram
        plt.title(f'Distribution of {var}')
        plt.xlabel(var)
        plt.ylabel('Frequency')
        plt.tight_layout()
        if not jupyter_render:
            plt.savefig(f"{filename}")
        else:
            plt.show()
        plt.close()

    def plot_scatter_two_vars(self, x, y, jupyter_render, filename="plot_two_vars.png"):
        plt.scatter(self.df[x].to_numpy(), self.df[y].to_numpy(), color='green')
        plt.title(f'Scatter Plot of {x} vs {y}')
        plt.xlabel(x)
        plt.ylabel(y)
        if not jupyter_render:
            plt.savefig(f"{filename}")
        else:
            plt.show()
        plt.close()

    def generate_report(self, title, name, graphs, variables=None, report_type="html"):
        if report_type == "pdf":
            self.generate_pdf_report(title, name, variables, graphs)
        else:
            profile = ProfileReport(self.df.select(variables).to_pandas(), title=title)
            profile.to_file(f"{name}.html")

    def generate_pdf_report(self, title, name, variables, graphs):
        pdf = FPDF()
        pdf.add_page()

        # Set the font
        try:
            pdf.set_font("Arial", size=12)
        except RuntimeError:
            pdf.set_font("Helvetica", size=12)  # Fallback font

        # Add title
        pdf.cell(200, 10, txt=title, ln=True, align="C")

        # Add summary statistics
        _stats = self.stats_summary(variables)

        # Convert Polars columns to lists
        statistics = _stats["statistic"].to_list()  # Get the 'statistic' column
        columns = _stats.columns[1:]  # Skip the 'statistic' column itself

        # Set column widths
        column_width = 22  # Reduced width for statistic values
        metric_width = 35  # Reduced width for the metric names
        pdf.set_font("Arial", size=10)

        # Headers (First column for metric names)
        pdf.cell(metric_width, 10, "Statistic", 1, 0, "C")
        for header in columns:  # Skip the first "statistic" column
            pdf.cell(column_width, 10, header, 1, 0, "C")
        pdf.ln()

        # Add data rows
        for i, stat_name in enumerate(statistics):  # Iterate over each statistic (row)
            pdf.cell(metric_width, 10, stat_name, 1, 0, "C")  # The statistic name (first column)

            # For each statistic, print the corresponding column values
            for column in columns:
                value = round(_stats[column][i], 4) if _stats[column][i] is not None else "N/A"
                pdf.cell(column_width, 10, str(value), 1, 0, "C")
            pdf.ln()

        # Add plots
        current_y = pdf.get_y() + 10  # Adjust starting y-position after the stats
        page_height = pdf.h - 20  # Page height minus margin

        for graph in graphs:
            if os.path.exists(graph):  # Check if image file exists
                # Check if adding the image would exceed the page height
                if current_y + 110 > page_height:  # Assuming image height is 110
                    pdf.add_page()  # Add new page
                    current_y = 10  # Reset Y position for new page

                pdf.image(graph, x=10, y=current_y, w=100)
                current_y += 110  # Adjust y position for the next image
            else:
                # Print error message if image file not found
                pdf.cell(200, 10, txt=f"Image {graph} not found.", ln=True)

        # Output to file
        pdf_output_path = f"{name}.pdf"
        pdf.output(pdf_output_path)

        print(f"PDF report generated: {pdf_output_path}")
