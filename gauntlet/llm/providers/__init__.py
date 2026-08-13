"""Vendor adapters. Import the concrete ones lazily - SDKs are optional at runtime."""

from gauntlet.llm.providers.scripted import ScriptedProvider

__all__ = ["ScriptedProvider"]
