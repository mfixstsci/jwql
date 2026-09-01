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
from urllib.parse import parse_qs, urlparse

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

# google AI dynamic UI example:
# 
# from urllib.parse import parse_qs, urlparse
# from shiny import App, render, ui
# 
# app_ui = ui.page_fluid(
#     ui.h2("Dynamic UI based on GET arguments"),
#     ui.output_ui("dynamic_content"),
# )
# 
# 
# def server(input, output, session):
#   @render.ui
#   def dynamic_content():
#     # Get the raw query string (e.g., "?theme=dark&user=alice")
#     query_string = session.clientdata.url_search()
# 
#     # Parse key-value pairs into a dictionary
#     parsed_params = parse_qs(urlparse(query_string).query)
#     user_type = parsed_params.get("type", ["guest"])[0]
# 
#     # Conditionally return different UI elements
#     if user_type == "admin":
#       return ui.div(
#           ui.h3("Welcome, Administrator!"),
#           ui.input_action_button("admin_btn", "Delete Database", class_="btn-danger"),
#       )
#     elif user_type == "premium":
#       return ui.div(
#           ui.h3("Welcome, Premium User!"),
#           ui.p("You have access to advanced features."),
#       )
#     else:
#       return ui.div(
#           ui.h3("Hello, Guest!"),
#           ui.p("Please log in to see more."),
#       )
# 
# 
# app = App(app_ui, server)


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

full_tabbed_ui = ui.navset_tab(
    ui.nav_menu(
        "MIRI",
        ui.nav_panel(
            "MIRI LRS Monitor",
            miri_lrs_ui
        ),
        ui.nav_panel(
            "MIRI MRS Monitor",
            ui.card(
                ui.card_header("MIRI MRS Card"),
            ),
        ),
    ),
    ui.nav_menu(
        "NIRCam",
        ui.nav_panel(
            "NIRCam TA Monitor",
            ui.card(
                ui.card_header("NIRCam Card"),
            ),
        ),
    ),
    ui.nav_menu(
        "NIRISS",
        ui.nav_panel(
            "NIRISS TA Monitor",
            ui.card(
                ui.card_header("NIRISS Card"),
            ),
        ),
    ),
    ui.nav_menu(
        "NIRSpec",
        ui.nav_panel(
            "NIRSpec TA Monitor",
            ui.card(
                ui.card_header("NIRSpec Card"),
            ),
        ),
    ),
)

app_ui = ui.page_fillable(
    ui.output_ui("dynamic_layout")
)

def server(input, output, session):
    @render.ui
    def dynamic_layout():
        if running_standalone:
            return full_tabbed_ui
        query_string = session.clientdata.url_search()
        parsed_params = parse_qs(urlparse(query_string).query)
        instrument = parsed_params.get("inst", ["unspecified"])[0]
        if instrument.lower() == "miri":
            return miri_lrs_ui
        elif instrument.lower() == "nircam":
            nircam_ui = ui.div(
                ui.h4("NIRCam Monitor"),
                ui.card(ui.card_header("NIRCam Card"))
            )
            return nircam_ui
        elif instrument.lower() == "niriss":
            niriss_ui = ui.div(
                ui.h4("NIRISS Monitor"),
                ui.card(ui.card_header("NIRISS Card"))
            )
            return niriss_ui
        elif instrument.lower() == "nirspec":
            nirspec_ui = ui.div(
                ui.h4("NIRSpec Monitor"),
                ui.card(ui.card_header("NIRSpec Card"))
            )
            return nirspec_ui
        else:
            return full_tabbed_ui
    @render.plot
    def plot_lrs_uncal_image():
        selected_data = uncal_data[
            input.group_slicer() - 1, input.integ_slicer() - 1, :, :
        ]
        fig = plt.imshow(selected_data, aspect='auto')
        plt.xlabel("x (pixels)", fontsize=11, fontweight="bold")
        plt.ylabel("y (pixels)", fontsize=11, fontweight="bold")
        cbar = plt.colorbar(fig, orientation="vertical", fraction=0.046, pad=0.04)
        cbar.set_label("Counts", fontsize=11, fontweight="bold")
        return fig
    @render.plot
    def plot_lrs_cal_image():
        fig = plt.imshow(cal_data, aspect='auto')
        plt.xlabel("x (pixels)", fontsize=11, fontweight="bold")
        plt.ylabel("y (pixels)", fontsize=11, fontweight="bold")
        cbar = plt.colorbar(fig, orientation="vertical", fraction=0.046, pad=0.04)
        cbar.set_label("Counts", fontsize=11, fontweight="bold")
        return fig
    @render.plot
    def plot_lrs_verification_image():
        fig = plt.imshow(cal_data, aspect='auto')
        plt.xlabel("x (pixels)", fontsize=11, fontweight="bold")
        plt.ylabel("y (pixels)", fontsize=11, fontweight="bold")
        cbar = plt.colorbar(fig, orientation="vertical", fraction=0.046, pad=0.04)
        cbar.set_label("Counts", fontsize=11, fontweight="bold")
        return fig
    @render.text
    def text_lrs_oss_log():
        query_string = session.clientdata.url_search()
        parsed_params = parse_qs(urlparse(query_string).query)
        instrument = parsed_params.get("inst", ["unspecified"])[0]
        scroll_text = f"Getting data for instrument {instrument} from {parsed_params}"
        return scroll_text

app = App(app_ui, server, debug=True)
