"""
bot_app.py

Entry point for the Adaptive UI bot (Bot Framework + aiohttp).

Run with:
    set PORT=3978
    set MicrosoftAppId=
    set MicrosoftAppPassword=
    python bot_app.py

Then connect the Bot Framework Emulator to:
    http://localhost:3978/api/messages
"""

import asyncio
import logging
import os

from aiohttp import web
from botbuilder.core import (
    BotFrameworkAdapterSettings,
    BotFrameworkAdapter,
    TurnContext,
)
from botbuilder.schema import Activity

from adaptive_bot import AdaptiveBot, UserStore

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("adaptive-ui-bot")

# Adapter & bot setup ---------------------------------------------------------

SETTINGS = BotFrameworkAdapterSettings(
    app_id=os.environ.get("MicrosoftAppId", ""),
    app_password=os.environ.get("MicrosoftAppPassword", ""),
)
ADAPTER = BotFrameworkAdapter(SETTINGS)

# Simple on-error handler so the Emulator shows useful messages
async def on_error(context: TurnContext, error: Exception):
    LOGGER.error("Bot error: %s", error, exc_info=True)
    await context.send_activity("Oops, something went wrong with the bot.")

ADAPTER.on_turn_error = on_error

# Instantiate our adaptive bot
BOT = AdaptiveBot(user_store=UserStore())

# aiohttp request handler ----------------------------------------------------

async def messages(request: web.Request) -> web.Response:
    """
    Main /api/messages endpoint for Bot Framework.
    """
    if "application/json" not in request.headers.get("Content-Type", ""):
        return web.Response(status=415)

    body = await request.json()
    activity = Activity().deserialize(body)

    auth_header = request.headers.get("Authorization", "")

    async def aux(turn_context: TurnContext):
        await BOT.on_turn(turn_context)

    await ADAPTER.process_activity(activity, auth_header, aux)
    return web.Response(status=201)

# Application bootstrap ------------------------------------------------------

def create_app() -> web.Application:
    app = web.Application()
    app.router.add_post("/api/messages", messages)
    return app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3978))
    app = create_app()
    LOGGER.info("Starting Adaptive UI bot on port %s ...", port)
    web.run_app(app, host="0.0.0.0", port=port)
