"""Migrations must match the models they claim to create.

Alembic only catches drift when it can reach a live database, which means a hand-written
migration can disagree with its model and nothing notices until deploy. These tests run
each migration's ``upgrade()`` against a recording stub in place of ``alembic.op``, then
compare what it would create against the SQLAlchemy metadata.

It does not replace running the migration for real, and it does not check that the SQL is
valid. What it does catch is the failure that actually happens: a column added to a model
and forgotten in the migration.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import sqlalchemy as sa

from gauntlet.db.models import Base

MIGRATIONS = Path(__file__).resolve().parents[2] / "alembic" / "versions"


@dataclass
class RecordedTable:
    name: str
    columns: dict[str, Any] = field(default_factory=dict)
    constraints: list[Any] = field(default_factory=list)


class RecordingOp:
    """Stands in for ``alembic.op``, recording DDL instead of emitting it."""

    def __init__(self) -> None:
        self.tables: dict[str, RecordedTable] = {}
        self.indexes: list[tuple[str, str, tuple[str, ...]]] = []
        self.executed: list[str] = []

    def create_table(self, name: str, *args: Any, **_: Any) -> None:
        table = RecordedTable(name=name)
        for item in args:
            if isinstance(item, sa.Column):
                table.columns[item.name] = item
            else:
                table.constraints.append(item)
        self.tables[name] = table

    def add_column(self, table: str, column: sa.Column, **_: Any) -> None:
        # Later migrations add columns to tables an earlier one created, so the recorded
        # shape has to accumulate rather than only reflect CREATE TABLE.
        self.tables.setdefault(table, RecordedTable(name=table)).columns[column.name] = column

    def drop_column(self, table: str, name: str, **_: Any) -> None:
        recorded = self.tables.get(table)
        if recorded is not None:
            recorded.columns.pop(name, None)

    def create_index(
        self, name: str, table: str, columns: list[str], **_: Any
    ) -> None:
        self.indexes.append((name, table, tuple(columns)))

    def execute(self, statement: Any) -> None:
        self.executed.append(str(statement))

    def f(self, name: str) -> str:
        return name

    def __getattr__(self, _name: str) -> Any:
        # Anything else a migration might call is irrelevant to schema shape.
        return lambda *args, **kwargs: None


def load(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(f"migration_{path.stem}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def in_revision_order() -> list[Any]:
    """Migration modules ordered by the revision chain, not by filename.

    Filenames sort by a random hex prefix, which has nothing to do with apply order. A
    migration that adds a column to a table an earlier one created has to run second or
    the recorded schema is wrong.
    """
    modules = {module.revision: module for module in map(load, MIGRATIONS.glob("*.py"))}
    children = {module.down_revision: rev for rev, module in modules.items()}

    ordered: list[Any] = []
    current = children.get(None)
    seen: set[str] = set()
    while current is not None and current not in seen:
        seen.add(current)
        ordered.append(modules[current])
        current = children.get(current)

    assert len(ordered) == len(modules), "revision chain does not reach every migration"
    return ordered


def all_recorded() -> RecordingOp:
    """Every migration applied in order against one accumulating schema."""
    recorder = RecordingOp()
    for module in in_revision_order():
        module.op = recorder
        module.upgrade()
    return recorder


class TestMigrationsMatchModels:
    def test_every_model_table_is_created(self):
        recorded = all_recorded()
        missing = set(Base.metadata.tables) - set(recorded.tables)
        assert not missing, f"models have tables no migration creates: {sorted(missing)}"

    def test_no_migration_creates_a_table_the_models_dropped(self):
        recorded = all_recorded()
        extra = set(recorded.tables) - set(Base.metadata.tables)
        assert not extra, f"migrations create tables no model defines: {sorted(extra)}"

    def test_every_model_column_exists_in_its_migration(self):
        recorded = all_recorded()
        problems: list[str] = []
        for name, table in Base.metadata.tables.items():
            created = recorded.tables.get(name)
            if created is None:
                continue
            for column in table.columns:
                if column.name not in created.columns:
                    problems.append(f"{name}.{column.name}")
        assert not problems, f"columns missing from migrations: {problems}"

    def test_nullability_matches(self):
        """A column nullable in one and not the other fails only on real data."""
        recorded = all_recorded()
        mismatched: list[str] = []
        for name, table in Base.metadata.tables.items():
            created = recorded.tables.get(name)
            if created is None:
                continue
            for column in table.columns:
                migrated = created.columns.get(column.name)
                if migrated is None:
                    continue
                if bool(column.nullable) != bool(migrated.nullable):
                    mismatched.append(
                        f"{name}.{column.name}: model nullable={column.nullable}, "
                        f"migration nullable={migrated.nullable}"
                    )
        assert not mismatched, mismatched

    def test_the_revision_chain_is_linear_and_complete(self):
        """One root, no forks, no dangling parents."""
        revisions = {
            module.revision: module.down_revision
            for module in map(load, MIGRATIONS.glob("*.py"))
        }

        roots = [rev for rev, parent in revisions.items() if parent is None]
        assert len(roots) == 1, f"expected one root revision, found {roots}"

        parents = [parent for parent in revisions.values() if parent is not None]
        assert len(parents) == len(set(parents)), "two migrations share a parent (fork)"
        for parent in parents:
            assert parent in revisions, f"revision {parent} is referenced but missing"
