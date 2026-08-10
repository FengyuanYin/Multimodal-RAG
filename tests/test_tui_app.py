import pytest

pytest.importorskip("textual")

from agentic_rag.tui.app import AutoMemoryApp
from agentic_rag.tui.paths import AutoMemoryPaths


@pytest.mark.asyncio
async def test_workspace_navigation(tmp_path):
    app = AutoMemoryApp(paths=AutoMemoryPaths.resolve(tmp_path / "home"))
    async with app.run_test(size=(140, 42)) as pilot:
        await pilot.pause()
        assert type(app.screen).__name__ == "ChatScreen"
        for key, expected in (("2", "KnowledgeScreen"), ("3", "EvaluationScreen"), ("4", "SettingsScreen"), ("5", "HelpScreen")):
            await pilot.press(key)
            await pilot.pause()
            assert type(app.screen).__name__ == expected
