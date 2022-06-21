from typing import NewType

from magicgui import register_type

weight = NewType("weight", float)

register_type(weight, widget_type="FloatSpinBox", max=1.0, value=1.0)
