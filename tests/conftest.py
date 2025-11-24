"""Pytest plugin to support async tests without external plugins."""

import asyncio
import inspect
from typing import Any

import pytest


@pytest.hookimpl
def pytest_configure(config: pytest.Config) -> None:
    """Register asyncio marker for async tests."""
    config.addinivalue_line(
        "markers", "asyncio: mark test to run in an asyncio event loop."
    )


def _resolve_fixture_kwargs(
    fixturedef: pytest.FixtureDef[Any], request: pytest.FixtureRequest
) -> dict[str, Any]:
    """Resolve the keyword arguments for a fixture function."""
    return {
        arg: request.getfixturevalue(arg)
        for arg in getattr(fixturedef, "argnames", []) or []
    }


@pytest.hookimpl(tryfirst=True)
def pytest_fixture_setup(
    fixturedef: pytest.FixtureDef[Any], request: pytest.FixtureRequest
) -> Any:
    """Handle async fixtures by running them in an event loop."""
    if inspect.isasyncgenfunction(fixturedef.func):
        async_gen = fixturedef.func(**_resolve_fixture_kwargs(fixturedef, request))

        async def _start() -> Any:
            return await async_gen.__anext__()

        result = asyncio.run(_start())

        def _finalizer() -> None:
            async def _finish() -> None:
                try:
                    await async_gen.__anext__()
                except StopAsyncIteration:
                    pass

            asyncio.run(_finish())

        request.addfinalizer(_finalizer)
        return result

    if inspect.iscoroutinefunction(fixturedef.func):
        coro = fixturedef.func(**_resolve_fixture_kwargs(fixturedef, request))
        return asyncio.run(coro)

    return None


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    """Execute async test functions using asyncio.run."""
    test_function = pyfuncitem.obj
    if inspect.iscoroutinefunction(test_function):
        signature = inspect.signature(test_function)
        kwargs = {
            name: pyfuncitem.funcargs[name]
            for name in signature.parameters.keys()
            if name in pyfuncitem.funcargs
        }
        asyncio.run(test_function(**kwargs))
        return True
    return None
