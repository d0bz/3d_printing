# coding=utf-8
from setuptools import find_packages, setup

plugin_identifier = "printguard"
plugin_package = "octoprint_printguard"
plugin_name = "OctoPrint-PrintGuard"
plugin_version = "0.1.0"
plugin_description = (
    "Hybrid AI & Computer Vision 3D print failure detection plugin for OctoPrint "
    "(detects air printing / no extrusion, bed detachment / warping, spaghetti, and nozzle blobs "
    "with automatic print pause and Discord/Telegram/Pushover/Webhook alerts)."
)
plugin_author = "d0bz"
plugin_author_email = ""
plugin_url = "https://github.com/d0bz/3d_printing"
plugin_license = "AGPLv3"

plugin_requires = [
    "requests>=2.25.0",
    "Pillow>=8.0.0",
    "numpy>=1.19.0",
]

plugin_additional_data = []
plugin_additional_packages = []
plugin_ignored_packages = []

setup(
    name=plugin_name,
    version=plugin_version,
    description=plugin_description,
    author=plugin_author,
    author_email=plugin_author_email,
    url=plugin_url,
    license=plugin_license,
    packages=find_packages(exclude=["tests", "tests.*"]),
    package_data={
        plugin_package: [
            "templates/*.jinja2",
            "static/js/*.js",
            "static/css/*.css",
            "static/*.apk",
        ]
    },
    include_package_data=True,
    install_requires=plugin_requires,
    extras_require={
        "cv": ["opencv-python-headless>=4.5.0", "onnxruntime>=1.14.0"],
        "dev": ["pytest>=7.0.0", "pytest-cov>=4.0.0"],
    },
    python_requires=">=3.8,<4",
    entry_points={
        "octoprint.plugin": [
            f"{plugin_identifier} = {plugin_package}"
        ]
    },
)
