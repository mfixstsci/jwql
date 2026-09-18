from shiny import App, module, reactive, render, ui
from shiny.types import SilentException

from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)

import matplotlib.pyplot as plt

from asgiref.sync import sync_to_async
from astropy.io import fits
from astroquery.mast import MastMissions
import numpy as np
import os
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from support_ta_monitor_data import TADataSupplier
from support_ta_monitor_logs import get_ictm_event_log
from support_ta_monitor_logs import extract_oss_event_msgs_for_visit
from support_ta_monitor_logs import check_log_and_note_issues

running_standalone = str(os.environ.get("SHINY_EMBED", 0)) == "0"

logging.info(f"SHINY_EMBED={os.environ.get('SHINY_EMBED')}")
logging.info(f"Running Standalone: {running_standalone}")

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"  # Optional: also bolds axis title

CANONICAL_NAMES = {
    "miri": "MIRI",
    "nircam": "NIRCam",
    "niriss": "NIRISS",
    "nirspec": "NIRSpec"
}

data_source = reactive.value(None)
current_instrument = reactive.value("")
current_mode = reactive.value("")
uncal_image = reactive.value("")
cal_image = reactive.value("")
check_image = reactive.value("")
user_connected = reactive.value(False)

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
        ui.input_selectize(
            "exposure_select",
            "Select Exposure",
            choices=[],
            selected=None,
            multiple=False,  # Set to True if you want a multi-tag text input
            width="500px",
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
                max_height="500px",
                full_screen=True
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
                max_height="500px",
                full_screen=True
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
                max_height="500px",
                full_screen=True
            ),
            ui.card(
                ui.card_header("OSS Log"),
                ui.output_ui("oss_warnings"),
                ui.div(
                    ui.output_code("text_miri_oss_log"),
                    style="font-size: 12px;"
                ),
                max_height="500px",
                full_screen=True
            ),
        ),
    )
    return miri_ui

@module.server
def miri_tab_server(input, output, session):
    oss_messages = reactive.value([])
    @render.ui
    def miri_uncal():
        return ui.card_header(f"TA Image (uncalibrated) {uncal_image()}"),
    @render.plot
    def plot_miri_uncal_image():
        selected_exposure = input.exposure_select()
        data_source().select_obs(selected_exposure)
        uncal_file = data_source().get_obs_uncal()
        if uncal_file is not None:
            uncal_image.set(Path(uncal_file).stem)
            with fits.open(uncal_file) as fits_file:
                uncal_data = fits_file['SCI'].data
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
        else:
            fig = plt.figure()
            fig.text(0.5, 0.5, 'No File Available', fontsize=18, ha='center', va='center')
        return fig
    @render.ui
    def miri_cal():
        return ui.card_header(f"TA Image (calibrated) {cal_image()}"),
    @render.plot
    def plot_miri_cal_image():
        selected_exposure = input.exposure_select()
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
        else:
            fig = plt.figure()
            fig.text(0.5, 0.5, 'No File Available', fontsize=18, ha='center', va='center')
        return fig
    @render.ui
    def miri_check():
        return ui.card_header(f"TA Image (check) {check_image()}"),
    @render.plot
    async def plot_miri_verification_image():
        selected_exposure = input.exposure_select()
        exp_data = data_source()
        exp_data.select_obs(selected_exposure)
        check_file = await sync_to_async(exp_data.get_obs_verification)()
        if check_file is not None:
            check_image.set(Path(check_file).stem)
            with fits.open(check_file) as fits_file:
                check_data = fits_file['SCI'].data
            fig = plt.imshow(check_data, aspect='auto', norm='log')
            plt.xlabel("x (pixels)", fontsize=11, fontweight="bold")
            plt.ylabel("y (pixels)", fontsize=11, fontweight="bold")
            cbar = plt.colorbar(fig, orientation="vertical", fraction=0.046, pad=0.04)
            cbar.set_label("Counts", fontsize=11, fontweight="bold")
        else:
            fig = plt.figure()
            fig.text(0.5, 0.5, 'No File Available', fontsize=18, ha='center', va='center')
        return fig
    @render.text
    def text_miri_oss_log():
        nonlocal oss_messages
        selected_exposure = input.exposure_select()
        data_source().select_obs(selected_exposure)
        uncal_file = data_source().get_obs_uncal()
        if uncal_file is not None:
            oss_messages.set([])
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
            all_warnings = []
            for i, message in enumerate(msgs):
                warning = check_log_and_note_issues(message)
                if warning is not None:
                    logging.info(f"Got OSS warning: {warning}")
                    all_warnings.append(f"Line {i+1}: {warning}")
            if len(all_warnings) > 0:
                oss_messages.set(all_warnings)
            return "\n".join(msgs)
    @render.ui
    def oss_warnings():
        nonlocal oss_messages
        if len(oss_messages()) > 0:
            message_text = []
            for message in oss_messages():
                message_text.append(ui.tags.div(f"Warning: {message}"))
            return ui.tags.div(
                *message_text,
                class_="alert alert-warning",
                role="alert"
            )
        return None

miri_ui = {
    "initial": "lrs",
    "panels": [
        build_nav_panel("LRS", miri_tab_ui("miri_lrs")),
        build_nav_panel("MRS", miri_tab_ui("miri_mrs"))
    ]
}

nircam_tab_ui = ui.card(
    ui.card_header("NIRCam Card")
)

nircam_ui = {
    "initial": "nircam",
    "panels": [
        build_nav_panel("NIRCam", nircam_tab_ui)
    ]
}

niriss_tab_ui = ui.card(
    ui.card_header("NIRISS Card")
)

niriss_ui = {
    "initial": "niriss",
    "panels": [
        build_nav_panel("NIRISS", niriss_tab_ui)
    ]
}

nirspec_tab_ui = ui.card(
    ui.card_header("NIRSpec Card")
)

nirspec_ui = {
    "initial": "nirspec",
    "panels": [
        build_nav_panel("NIRSpec", nirspec_tab_ui)
    ]
}


instrument_ui = {
    "miri": miri_ui,
    "nircam": nircam_ui,
    "niriss": niriss_ui,
    "nirspec": nirspec_ui,
}

app_ui = ui.page_navbar(
    title=ui.output_ui("dynamic_title"),
    id="nav_toplevel",
    fillable=True
)

def server(input, output, session):
    tab_setup = False

    miri_tab_server("miri_lrs")

    miri_tab_server("miri_mrs")

    @render.ui
    def dynamic_title():
        selected_instrument = current_instrument()
        if selected_instrument in CANONICAL_NAMES:
            display_name = CANONICAL_NAMES[selected_instrument]
            logging.info(f"Updating navbar title to {display_name}")
            return ui.span(display_name, class_="navbar-brand")
        return ui.span("", class_="navbar-brand")

    def set_exposure_options(instrument, mode, options):
        logging.info("Running set_exposure_options")
        selectize_id = f"{instrument}_{mode}-exposure_select"
        logging.info(f"Selectize ID is {selectize_id}")
        ui.update_selectize(
            selectize_id,
            choices=options,
            selected=options[0],
        )
        logging.info("Finished set_exposure_options")

    @reactive.effect
    @reactive.event(input.nav_toplevel)
    async def _():
        mode = input.nav_toplevel()
        logging.info("Running effect based on tab changing")
        instrument = current_instrument()
        if instrument is not None and mode is not None:
            mode = mode.lower()
            logging.info(f"Instrument and mode are {instrument}, {mode}")
            if data_source() is None or data_source().instrument != instrument or data_source().mode != mode:
                logging.info(f"Creating data source for {instrument} {mode}")
                data_source.set(TADataSupplier(instrument, mode))
                logging.info(f"Data source created")
            logging.info("Getting exposure list")
            obs_data = data_source()
            obs_list = await sync_to_async(obs_data.get_obs_list)()
            logging.info(f"Putting {len(obs_list)} exposures in select")
            set_exposure_options(instrument, mode, obs_list)
            

    @reactive.effect
    def _():
        nonlocal tab_setup
        if not tab_setup:
            query_string = session.clientdata.url_search()
            parsed_params = parse_qs(urlparse(query_string).query)
            if "inst" in parsed_params:
                instrument = parsed_params["inst"][0].lower()
                logging.info(f"Got instrument {instrument}")
                if instrument in CANONICAL_NAMES:
                    for panel in instrument_ui[instrument]["panels"]:
                        ui.insert_nav_panel(
                            id="nav_toplevel",
                            nav_panel=panel,
                            select=False
                        )
                    ui.update_navset("nav_toplevel", selected=instrument_ui[instrument]["initial"])
                    current_instrument.set(instrument)
                    tab_setup = True
                    return
            # Currently, just make MIRI either way.
            instrument = "miri"
            ui.update_page_title(f"{CANONICAL_NAMES[instrument]}")
            current_instrument.set(instrument)
            for panel in instrument_ui[instrument]["panels"]:
                ui.insert_nav_panel(
                    id="nav_toplevel",
                    nav_panel=panel,
                    select=False
                )
            ui.update_navset("nav_toplevel", selected=instrument_ui[instrument]["initial"])
            tab_setup = True

app = App(app_ui, server, debug=False)
