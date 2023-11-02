from pathlib import Path
from typing import Annotated

from magicclass import (
    MagicTemplate,
    do_not_record,
    magicclass,
    magictoolbar,
    nogui,
    set_design,
    set_options,
)
from magicgui.widgets import Table


@magicclass(widget_type="tabbed")
class TableStack(MagicTemplate):
    @magictoolbar
    class Tools(MagicTemplate):
        def _get_current_index(self, w=None) -> int:
            parent = self.find_ancestor(TableStack)
            return parent.current_index

        @set_design(icon="fluent-emoji-high-contrast:open-file-folder")
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
        @set_design(icon="material-symbols:save")
        def save(
            self, path: Path, idx: Annotated[int, {"bind": _get_current_index}]
        ):
            """Save current table to csv."""
            table: Table = self.find_ancestor(TableStack)[idx]
            df = table.to_dataframe()
            df.to_csv(path, index=False)

        @set_design(icon="ph:copy-bold")
        def copy(self, idx: Annotated[int, {"bind": _get_current_index}]):
            """Copy current table to clipboard."""
            table: Table = self.find_ancestor(TableStack)[idx]
            df = table.to_dataframe()
            df.to_clipboard(excel=True, index=False)

        @set_design(icon="mdi:trash-outline")
        def delete(self, idx: Annotated[int, {"bind": _get_current_index}]):
            """Delete current table"""
            del self.find_ancestor(TableStack)[idx]

    @nogui
    @do_not_record
    def add_table(self, df, name: str = None):
        if name is None:
            name = f"Table-{len(self)}"

        table = Table(value=df, name=self._coerce_name(name))
        table.read_only = True
        self.append(table)

    def _coerce_name(self, name: str):
        stem = name
        suffix = 0
        while name in self:
            name = f"{stem}-{suffix}"
            suffix += 1
        return name
