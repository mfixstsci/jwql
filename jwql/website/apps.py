#! /usr/bin/env python

"""
apps.py is the standard and recommended way to configure application-specific settings
in Django, including tasks like importing additional modules during initialization.

Author
------

B. Hilbert
"""

from django.apps import AppConfig


class JwqlAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'jwql'

    def ready(self):
        # Import models not defined in models.py here
        # By importing these models here, they will be available
        # to the build_table() function.
        import jwql.website.apps.jwql.monitor_models.bad_pixel
        import jwql.website.apps.jwql.monitor_models.bias
        import jwql.website.apps.jwql.monitor_models.claw
        import jwql.website.apps.jwql.monitor_models.common
        import jwql.website.apps.jwql.monitor_models.dark_current
        import jwql.website.apps.jwql.monitor_models.readnoise
        import jwql.website.apps.jwql.monitor_models.ta
        import jwql.website.apps.jwql.monitor_models.wisp_finder
