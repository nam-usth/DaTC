# Linux release only: the bundled fontconfig looks for its configuration in the build machine's
# environment; point it at the system's file unless the user has set one.
import os

if not os.environ.get("FONTCONFIG_FILE") and not os.environ.get("FONTCONFIG_PATH") and os.path.exists("/etc/fonts/fonts.conf"):
    os.environ["FONTCONFIG_FILE"] = "/etc/fonts/fonts.conf"
