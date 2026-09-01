from shiny import App, reactive, render, ui

# from shiny.express import app_opts, input, render, session, ui
# from shiny import reactive
import logging

logging.basicConfig(level=logging.DEBUG)
# app_opts(debug=False)

import matplotlib.pyplot as plt

from astropy.io import fits
from astroquery.mast import MastMissions
import numpy as np
import os

running_standalone = str(os.environ.get("SHINY_EMBED", 0)) == "0"

logging.info(f"SHINY_EMBED={os.environ.get('SHINY_EMBED')}")
logging.info(f"Running Standalone: {running_standalone}")

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"  # Optional: also bolds axis title

url_arguments = reactive.value("")
@render.code
def get_url_arguments():
    url_arguments.set(session.clientdata.url_search())
    logging.info(f"URL search values are: {url_arguments.get()}")

# Uncal data
# Two groups, 4 integrations
rng = np.random.default_rng()
uncal_data = rng.random((2, 4, 1024, 1032))
n_groups, n_ints, _, _ = uncal_data.shape

# Calibrated data
cal_data = rng.random((1024, 1032))

missions = MastMissions(mission="jwst")
obs_table = missions.query_criteria(instrume="MIRI", exp_type="MIR_TA*")

# with ui.navset_tab(id="my_tabs"):
#     with ui.nav_menu("MIRI"):
#         with ui.nav_panel("MIRI LRS Monitor"):
#             if running_standalone:
#                 ui.h4("MIRI LRS Quick Look")
#             with ui.div():
#                 ui.input_selectize(
#                     "miri_lrs_fileset_select",
#                     "Select MIRI LRS TA Exposure",
#                     choices=obs_table["fileSetName"].tolist(),
#                     selected=None,
#                     multiple=False,  # Set to True if you want a multi-tag text input
#                     options={
#                         "placeholder": "Enter FileSetName",
#                         "create": True,  # Allows typing custom values not in the list
#                         "persist": False,  # User-created choices don't permanently alter the original list
#                         "openOnFocus": True,  # Opens dropdown immediately when clicked
#                         "allowEmptyOption": True,
#                     },
#                 )
#                 with ui.layout_columns(col_widths=[6, 6]):
#                     with ui.card(full_screen=True):
#                         ui.card_header("TAimage_uncal")
#                         n_groups, n_ints, _, _ = uncal_data.shape
#                         with ui.div(
#                             class_="d-flex justify-content-center align-items-center w-100"
#                         ):
#                             ui.input_slider(
#                                 "group_slicer",
#                                 "Uncal Groups:",
#                                 min=1,
#                                 max=n_groups,
#                                 value=1,
#                                 step=1,
#                             )
# 
#                         with ui.div(
#                             class_="d-flex justify-content-center align-items-center w-100"
#                         ):
#                             ui.input_slider(
#                                 "integ_slicer",
#                                 "Uncal Integrations:",
#                                 min=1,
#                                 max=n_ints,
#                                 value=1,
#                                 step=1,
#                             )
# 
#                         @render.plot
#                         def plot_lrs_uncal_image():
#                             selected_data = uncal_data[
#                                 input.group_slicer() - 1, input.integ_slicer() - 1, :, :
#                             ]
#                             fig = plt.imshow(selected_data)
#                             plt.xlabel("x (pixels)", fontsize=11, fontweight="bold")
#                             plt.ylabel("y (pixels)", fontsize=11, fontweight="bold")
#                             cbar = plt.colorbar(fig, orientation="horizontal", pad=0.15)
#                             cbar.set_label("Counts", fontsize=11, fontweight="bold")
#                             return fig
# 
#                     with ui.card(full_screen=True):
#                         ui.card_header("TAimage_cal")
# 
#                         @render.plot
#                         def plot_lrs_cal_image():
#                             fig = plt.imshow(cal_data)
#                             plt.xlabel("x (pixels)", fontsize=11, fontweight="bold")
#                             plt.ylabel("y (pixels)", fontsize=11, fontweight="bold")
#                             cbar = plt.colorbar(fig, orientation="horizontal", pad=0.15)
#                             cbar.set_label("Counts", fontsize=11, fontweight="bold")
#                             return fig
# 
#                 with ui.layout_columns(col_widths=[6, 6]):
#                     with ui.card(full_screen=True):
#                         ui.card_header("TA Verification Image")
# 
#                         @render.plot
#                         def plot_lrs_verfication_img():
#                             fig = plt.imshow(cal_data)
#                             plt.xlabel("x (pixels)", fontsize=11, fontweight="bold")
#                             plt.ylabel("y (pixels)", fontsize=11, fontweight="bold")
#                             cbar = plt.colorbar(fig, orientation="horizontal", pad=0.15)
#                             cbar.set_label("Counts", fontsize=11, fontweight="bold")
#                             return fig
# 
#                     with ui.card(max_height="500px", full_screen=True):
#                         ui.card_header("OSS Log")
# 
#                         @render.text
#                         def lrs_text():
#                             scroll_text = """Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum
#                             Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum
#                             Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum
#                             Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum
#                             Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum
#                             Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum
#                             Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum
#                             Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum
#                             Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum
#                             Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum
#                             Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum
#                             Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum"""
#                             return scroll_text
# 
#         with ui.nav_panel("MIRI MRS Monitor"):
#             ui.h4("MIRI MRS Quick Look")
#             with ui.div():
#                 with ui.layout_columns(col_widths=[6, 6]):
#                     with ui.card(full_screen=True):
#                         ui.card_header("Card 1")
#                     with ui.card(full_screen=True):
#                         ui.card_header("Card 2")
# 
#     with ui.nav_menu("NIRCam"):
#         with ui.nav_panel("NIRCam TA Monitor"):
#             ui.h4("NIRCam TA Monitor")
#             with ui.div():
#                 with ui.layout_columns(col_widths=[6, 6]):
#                     with ui.card(full_screen=True):
#                         ui.card_header("Card 1")
# 
#                     with ui.card(full_screen=True):
#                         ui.card_header("Card 2")
# 
#     with ui.nav_menu("NIRSpec"):
#         with ui.nav_panel("NIRSpec TA Monitor"):
#             ui.h4("NIRSpec TA Monitor")
#             with ui.div():
#                 with ui.layout_columns(col_widths=[6, 6]):
#                     with ui.card(full_screen=True):
#                         ui.card_header("Card 1")
# 
#                     with ui.card(full_screen=True):
#                         ui.card_header("Card 2")
# 
#     with ui.nav_menu("NIRISS"):
#         with ui.nav_panel("NIRISS TA Monitor"):
#             ui.h4("NIRISS TA Monitor")
#             with ui.div():
#                 with ui.layout_columns(col_widths=[6, 6]):
#                     with ui.card(full_screen=True):
#                         ui.card_header("Card 1")
# 
#                     with ui.card(full_screen=True):
#                         ui.card_header("Card 2")

miri_lrs_ui = ui.div(
    ui.h4("MIRI LRS"),
    ui.input_selectize(
        "miri_lrs_fileset_select",
        "Select MIRI LRS TA Exposure",
        choices=obs_table["fileSetName"].tolist(),
        selected=None,
        multiple=False,  # Set to True if you want a multi-tag text input
        options={
            "placeholder": "Enter FileSetName",
            "create": True,  # Allows typing custom values not in the list
            "persist": False,  # User-created choices don't permanently alter the original list
            "openOnFocus": True,  # Opens dropdown immediately when clicked
            "allowEmptyOption": True,
        },
    ),
    ui.layout_columns(
        ui.card(
            ui.card_header("TA Image (uncalibrated)"),
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_slider(
                        "group_slicer",
                        "Uncal Groups:",
                        min=1,
                        max=n_groups,
                        value=1,
                        step=1,
                    ),
                    ui.input_slider(
                        "integ_slicer",
                        "Uncal Integrations:",
                        min=1,
                        max=n_ints,
                        value=1,
                        step=1,
                    ),
                    "Group and Integration Selection",
                    open="closed",
                ),
            ),
            ui.output_plot("plot_lrs_uncal_image"),#, width="100%", height="400px"),
            max_height="500px"
        ),
        ui.card(
            ui.card_header("TA Image (calibrated)"),
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_checkbox(
                        "check_lrs_show_calibrated_crosses",
                        "Show TA checks",
                        True
                    ),
                    "Display Selection",
                    open="closed",
                ),
            ),
            ui.output_plot("plot_lrs_cal_image"),#, width="100%", height="400px"),
            max_height="500px"
        ),
    ),
    ui.layout_columns(
        ui.card(
            ui.card_header("TA Verification Image"),
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_checkbox(
                        "check_lrs_show_verification_crosses",
                        "Show TA checks",
                        True
                    ),
                    "Display Selection",
                    open="closed",
                ),
            ),
            ui.output_plot("plot_lrs_verification_image"),#, width="100%", height="400px"),
            max_height="500px"
        ),
        ui.card(
            ui.card_header("OSS Log"),
            ui.output_text("text_lrs_oss_log"),
            max_height="500px",
        ),
    ),
)

app_ui = ui.page_fillable(
    miri_lrs_ui
)

def server(input, output, session):
    @render.plot
    def plot_lrs_uncal_image():
        selected_data = uncal_data[
            input.group_slicer() - 1, input.integ_slicer() - 1, :, :
        ]
        fig = plt.imshow(selected_data)
        plt.xlabel("x (pixels)", fontsize=11, fontweight="bold")
        plt.ylabel("y (pixels)", fontsize=11, fontweight="bold")
        cbar = plt.colorbar(fig, orientation="vertical", pad=0.15)
        cbar.set_label("Counts", fontsize=11, fontweight="bold")
        return fig
    @render.plot
    def plot_lrs_cal_image():
        fig = plt.imshow(cal_data)
        plt.xlabel("x (pixels)", fontsize=11, fontweight="bold")
        plt.ylabel("y (pixels)", fontsize=11, fontweight="bold")
        cbar = plt.colorbar(fig, orientation="vertical", pad=0.15)
        cbar.set_label("Counts", fontsize=11, fontweight="bold")
        return fig
    @render.plot
    def plot_lrs_verification_image():
        fig = plt.imshow(cal_data)
        plt.xlabel("x (pixels)", fontsize=11, fontweight="bold")
        plt.ylabel("y (pixels)", fontsize=11, fontweight="bold")
        cbar = plt.colorbar(fig, orientation="vertical", pad=0.15)
        cbar.set_label("Counts", fontsize=11, fontweight="bold")
        return fig
    @render.text
    def text_lrs_oss_log():
        scroll_text = """Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum
        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum
        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum
        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum
        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum
        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum
        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum
        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum
        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum
        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum
        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum
        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum"""
        return scroll_text

app = App(app_ui, server, debug=True)
