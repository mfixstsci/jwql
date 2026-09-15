from shiny import App, module, reactive, render, ui

from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)

import matplotlib.pyplot as plt

from astropy.io import fits
from astroquery.mast import MastMissions
import numpy as np
import os
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from support_ta_monitor_data import TADataSupplier

from log_msg_extraction import get_ictm_event_log, extract_oss_event_msgs_for_visit


running_standalone = str(os.environ.get("SHINY_EMBED", 0)) == "0"

logging.info(f"SHINY_EMBED={os.environ.get('SHINY_EMBED')}")
logging.info(f"Running Standalone: {running_standalone}")

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"  # Optional: also bolds axis title

data_source = reactive.value(None)
uncal_image = reactive.value("")
cal_image = reactive.value("")
check_image = reactive.value("")

def build_nav_panel(panel_name, panel_ui):
    return ui.nav_panel(panel_name, panel_ui)

def build_menu_ui(name, ui_list):
    nav_panels = [build_nav_panel(n, u) for n, u in ui_list]
    return ui.nav_menu(name, *nav_panels)

def build_navset_ui(menu_list, id="toplevel"):
    return ui.navset_tab(*menu_list, id=id)

@module.ui
def miri_tab_ui():
    miri_ui = ui.div(
        ui.output_ui("miri_title"),
        ui.input_selectize(
            "miri_exposure_select",
            "Select MIRI LRS TA Exposure",
            choices=[],
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
                ui.output_ui("miri_uncal"),
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.input_slider(
                            "group_slicer",
                            "Uncal Groups:",
                            min=1,
                            max=1,
                            value=1,
                            step=1,
                        ),
                        ui.input_slider(
                            "integ_slicer",
                            "Uncal Integrations:",
                            min=1,
                            max=1,
                            value=1,
                            step=1,
                        ),
                        open="closed",
                    ),
                ),
                ui.output_plot("plot_miri_uncal_image"),#, width="100%", height="400px"),
                max_height="500px"
            ),
            ui.card(
                ui.output_ui("miri_cal"),
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.input_checkbox(
                            "check_miri_show_calibrated_crosses",
                            "Show TA checks",
                            True
                        ),
                        open="closed",
                    ),
                ),
                ui.output_plot("plot_miri_cal_image"),#, width="100%", height="400px"),
                max_height="500px"
            ),
        ),
        ui.layout_columns(
            ui.card(
                ui.output_ui("miri_check"),
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.input_checkbox(
                            "check_miri_show_verification_crosses",
                            "Show TA checks",
                            True
                        ),
                        open="closed",
                    ),
                ),
                ui.output_plot("plot_miri_verification_image"),#, width="100%", height="400px"),
                max_height="500px"
            ),
            ui.card(
                ui.card_header("OSS Log"),
                ui.output_text("text_miri_oss_log"),
                max_height="500px",
            ),
        ),
    )
    return miri_ui

@module.server
def miri_tab_server(input, output, session):
    @render.ui
    def miri_title():
        return ui.h4(f"{input.miri_mode()}")
    @render.ui
    def miri_uncal():
        return ui.card_header(f"TA Image (uncalibrated) {uncal_image()}"),
    @render.plot
    def plot_miri_uncal_image():
        selected_exposure = input.miri_exposure_select()
        data_source().select_obs(selected_exposure)
        uncal_file = data_source().get_obs_uncal()
        if uncal_file is not None:
            uncal_image.set(Path(uncal_file).stem)
            with fits.open(uncal_file) as fits_file:
                uncal_data = fits_file['SCI'].data
            print(uncal_data.shape)
            ui.update_slider("group_slicer", min=1, max=uncal_data.shape[0])
            ui.update_slider("integ_slicer", min=1, max=uncal_data.shape[1])
            selected_data = uncal_data[
                input.group_slicer() - 1, input.integ_slicer() - 1, :, :
            ]
            fig = plt.imshow(selected_data, aspect='auto', norm='log')
            plt.xlabel("x (pixels)", fontsize=11, fontweight="bold")
            plt.ylabel("y (pixels)", fontsize=11, fontweight="bold")
            cbar = plt.colorbar(fig, orientation="vertical", fraction=0.046, pad=0.04)
            cbar.set_label("Counts", fontsize=11, fontweight="bold")
            return fig
    @render.ui
    def miri_cal():
        return ui.card_header(f"TA Image (calibrated) {cal_image()}"),
    @render.plot
    def plot_miri_cal_image():
        selected_exposure = input.miri_exposure_select()
        data_source().select_obs(selected_exposure)
        cal_file = data_source().get_obs_cal()
        if cal_file is not None:
            cal_image.set(Path(cal_file).stem)
            with fits.open(cal_file) as fits_file:
                cal_data = fits_file['SCI'].data
            fig = plt.imshow(cal_data, aspect='auto', norm='log')
            plt.xlabel("x (pixels)", fontsize=11, fontweight="bold")
            plt.ylabel("y (pixels)", fontsize=11, fontweight="bold")
            cbar = plt.colorbar(fig, orientation="vertical", fraction=0.046, pad=0.04)
            cbar.set_label("Counts", fontsize=11, fontweight="bold")
            return fig
    @render.ui
    def miri_check():
        return ui.card_header(f"TA Image (check) {check_image()}"),
    @render.plot
    def plot_miri_verification_image():
        selected_exposure = input.miri_exposure_select()
        data_source().select_obs(selected_exposure)
        check_file = data_source().get_obs_verification()
        if check_file is not None:
            check_image.set(Path(check_file).stem)
            with fits.open(check_file) as fits_file:
                check_data = fits_file['SCI'].data
            fig = plt.imshow(check_data, aspect='auto', norm='log')
            plt.xlabel("x (pixels)", fontsize=11, fontweight="bold")
            plt.ylabel("y (pixels)", fontsize=11, fontweight="bold")
            cbar = plt.colorbar(fig, orientation="vertical", fraction=0.046, pad=0.04)
            cbar.set_label("Counts", fontsize=11, fontweight="bold")
            return fig
    @render.text
    def text_miri_oss_log():
        selected_exposure = input.miri_exposure_select()
        data_source().select_obs(selected_exposure)
        uncal_file = data_source().get_obs_uncal()
        if uncal_file is not None:
            with fits.open(uncal_file) as fits_file:
                uncal_hdr = fits_file[0].header

            # Take an hour off of time to capture 
            startdate = datetime.fromisoformat(uncal_hdr["DATE-BEG"]) - timedelta(days=1)
            enddate = datetime.fromisoformat(uncal_hdr["DATE-END"])
            visit_id = uncal_hdr["VISIT_ID"]

            eventlog = get_ictm_event_log(
                        mast_api_token=None,
                        verbose=False,
                        startdate=startdate,
                        enddate=enddate,
                    )

            msgs = extract_oss_event_msgs_for_visit(eventlog, visit_id)

            return " ".join(msgs)

miri_ui = build_menu_ui(
    "MIRI",
    [("MIRI LRS", miri_tab_ui("miri_lrs")), ("MIRI MRS", miri_tab_ui("miri_mrs"))],
)

nircam_ui = ui.card(
    ui.card_header("NIRCam Card")
)

niriss_ui = ui.card(
    ui.card_header("NIRISS Card")
)

nirspec_ui = ui.card(
    ui.card_header("NIRSpec Card")
)



instrument_ui = {
    "miri": miri_ui,
    "nircam": build_menu_ui("NIRCam", [("NIRCam TA Monitor", nircam_ui)]),
    "niriss": build_menu_ui("NIRISS", [("NIRISS TA Monitor", niriss_ui)]),
    "nirspec": build_menu_ui("NIRSPEC", [("NIRSpec TA Monitor", nirspec_ui)]),
}

app_ui = ui.page_fillable(
    ui.output_ui("dynamic_layout")
)

def server(input, output, session):
    miri_tab_server("miri_lrs")
    miri_tab_server("miri_mrs")
    @render.ui
    def dynamic_layout():
        # Note that at some point we will need to update the data supplier based on the
        # currently selected tab
        logging.info("Creating data source")
        data_source.set(TADataSupplier("MIRI"))
        logging.info("Updating exposure select list")
        ui.update_selectize(
            "miri_lrs-miri_exposure_select",
            choices = data_source().obs_list
        )
        logging.info("Checking run mode")
        if running_standalone:
            return build_navset_ui([instrument_ui[x] for x in sorted(instrument_ui.keys())])
        query_string = session.clientdata.url_search()
        parsed_params = parse_qs(urlparse(query_string).query)
        instrument = parsed_params.get("inst", ["unspecified"])[0]
        if instrument.lower() in instrument_ui.keys():
            return build_navset_ui([instrument_ui[instrument.lower()]], id=f"{instrument.lower()}_title")
        else:
            return build_navset_ui([instrument_ui[x] for x in sorted(instrument_ui.keys())])

app = App(app_ui, server, debug=False)
