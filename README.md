# napari-filaments

[![License BSD-3](https://img.shields.io/pypi/l/napari-filaments.svg?color=green)](https://github.com/hanjinliu/napari-filaments/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-filaments.svg?color=green)](https://pypi.org/project/napari-filaments)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-filaments.svg?color=green)](https://python.org)
[![tests](https://github.com/hanjinliu/napari-filaments/workflows/tests/badge.svg)](https://github.com/hanjinliu/napari-filaments/actions)
[![codecov](https://codecov.io/gh/hanjinliu/napari-filaments/branch/main/graph/badge.svg)](https://codecov.io/gh/hanjinliu/napari-filaments)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-filaments)](https://napari-hub.org/plugins/napari-filaments)

A napari plugin for filament analysis.

This plugin helps you to manually track filaments using path shapes of `Shapes` layer.

![](https://github.com/hanjinliu/napari-filaments/raw/main/resources/fit.gif)

Currently implemented functions

- Fit paths to filaments in an image as a 2-D spline curve.
- Clip/extend paths.
- Measurement of filament length at sub-pixel precision.
- Basic quantification (mean, std, etc.) along paths.
- Import paths from ImageJ ROI file.

Basic Usage
-----------

Click `Layers > open image` to open a tif file. You'll find the image you chose and a shapes layer are added to the layer list.

![](https://github.com/hanjinliu/napari-filaments/raw/main/resources/fig.png)

- The "target filaments" box shows the working shapes layer.
- The "target image" box shows the image layer on which fitting and quantification will be conducted.
- The "filament" box shows currently selected shape (initially this box is empty).

Add path shapes and push key `F1` to fit the shape to the filament in the target image layer.


----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.


## Installation

You can install `napari-filaments` via [pip]:

    pip install napari-filaments



To install latest development version :

    pip install git+https://github.com/hanjinliu/napari-filaments.git


## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-filaments" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/hanjinliu/napari-filaments/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
