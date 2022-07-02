from pathlib import Path

from magicclass import (
    Icon,
    MagicTemplate,
    do_not_record,
    magicclass,
    magictoolbar,
    nogui,
    set_design,
    set_options,
)
from magicclass.types import Bound
from magicgui.widgets import Table


@magicclass(widget_type="tabbed")
class TableStack(MagicTemplate):
    @magictoolbar
    class Tools(MagicTemplate):
        def _get_current_index(self, w=None) -> int:
            parent = self.find_ancestor(TableStack)
            return parent.current_index

        @set_design(icon=Icon.DialogOpenButton)
        def open(self, path: Path):
            """Open a csv file."""
            import pandas as pd

            path = Path(path)
            df = pd.read_csv(path)
            table = Table(value=df, name=path.stem)
            table.read_only = True
            parent = self.find_ancestor(TableStack)
            parent.append(table)
            parent.current_index = len(parent) - 1

        @set_options(path={"mode": "w", "filter": "*.csv;*.txt"})
        @set_design(icon=Icon.DialogSaveButton)
        def save(self, path: Path, idx: Bound[_get_current_index]):
            """Save current table to csv."""
            table: Table = self.find_ancestor(TableStack)[idx]
            df = table.to_dataframe()
            df.to_csv(path, index=False)

        @set_design(icon=Icon.FileLinkIcon)
        def copy(self, idx: Bound[_get_current_index]):
            """Copy current table to clipboard."""
            table: Table = self.find_ancestor(TableStack)[idx]
            df = table.to_dataframe()
            df.to_clipboard(excel=True, index=False)

        @set_design(icon=Icon.DialogDiscardButton)
        def delete(self, idx: Bound[_get_current_index]):
            """Delete current table"""
            del self.find_ancestor(TableStack)[idx]

    @nogui
    @do_not_record
    def add_table(self, df, name: str = None):
        if name is None:
            name = f"Table-{len(self)}"
        table = Table(value=df, name=name)
        table.read_only = True
        self.append(table)
