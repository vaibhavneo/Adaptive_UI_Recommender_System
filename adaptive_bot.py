"""
adaptive_bot.py

Adaptive UI chatbot for AI-HCI assignment.

Key ideas:
- Tracks simple user model in memory (experience_level, errors, preference).
- Adapts responses AND card layout based on that user model:
  * New / overwhelmed users → step-by-step guidance, fewer choices.
  * Confident / expert users → compact answers, more options per card.
  * “Dark mode” preference → changes card titles and text to reflect the mode.
"""

from typing import Dict, Any
from dataclasses import dataclass, field

from botbuilder.core import ActivityHandler, TurnContext, MessageFactory
from botbuilder.schema import (
    Activity,
    Attachment,
    CardAction,
    ActionTypes,
    HeroCard,
)


# ---------- Simple in-memory user model ----------

@dataclass
class UserProfile:
    experience: str = "new"        # "new", "intermediate", "expert"
    errors: int = 0
    pref_mode: str = "guided"      # "guided" or "expert"
    theme: str = "light"           # "light" or "dark"


class UserStore:
    """
    Extremely simple in-memory user profile store.
    For a real system you would plug in CosmosDB, Redis, or Bot Framework state.
    """
    def __init__(self) -> None:
        self._store: Dict[str, UserProfile] = {}

    def get(self, user_id: str) -> UserProfile:
        if user_id not in self._store:
            self._store[user_id] = UserProfile()
        return self._store[user_id]

    def update(self, user_id: str, **kwargs: Any) -> UserProfile:
        profile = self.get(user_id)
        for k, v in kwargs.items():
            if hasattr(profile, k):
                setattr(profile, k, v)
        return profile


# ---------- Adaptive bot implementation ----------

class AdaptiveBot(ActivityHandler):
    def __init__(self, user_store: UserStore | None = None):
        super().__init__()
        self._users = user_store or UserStore()

    # ---- Helpers to build “adaptive UI” HeroCards ----

    def _build_guided_card(self, profile: UserProfile) -> Attachment:
        """
        Beginner / guided mode: one main action and clear explanation.
        """
        title = "Step-by-step help"
        subtitle = "I’ll guide you through one decision at a time."
        if profile.theme == "dark":
            title = "Step-by-step help (Dark Mode)"

        card = HeroCard(
            title=title,
            subtitle=subtitle,
            text=(
                "You can ask questions in natural language.\n\n"
                "Try:\n"
                "• \"Show me beginner tips\"\n"
                "• \"Switch to expert mode\""
            ),
            buttons=[
                CardAction(
                    type=ActionTypes.im_back,
                    title="Beginner tips",
                    value="beginner tips",
                ),
                CardAction(
                    type=ActionTypes.im_back,
                    title="Switch to expert mode",
                    value="expert mode",
                ),
            ],
        )
        return Attachment(
            content_type="application/vnd.microsoft.card.hero",
            content=card,
        )

    def _build_expert_card(self, profile: UserProfile) -> Attachment:
        """
        Expert mode: denser card with several quick actions.
        """
        title = "Power user panel"
        subtitle = "Short, dense options for experienced users."
        if profile.theme == "dark":
            title = "Power user panel (Dark Mode)"

        card = HeroCard(
            title=title,
            subtitle=subtitle,
            text="Choose a shortcut, or just type a command.",
            buttons=[
                CardAction(
                    type=ActionTypes.im_back,
                    title="Show all commands",
                    value="help",
                ),
                CardAction(
                    type=ActionTypes.im_back,
                    title="Explain recommendations",
                    value="explain recommender",
                ),
                CardAction(
                    type=ActionTypes.im_back,
                    title="Switch to guided mode",
                    value="guided mode",
                ),
            ],
        )
        return Attachment(
            content_type="application/vnd.microsoft.card.hero",
            content=card,
        )

    # ---- Main message handler ----

    async def on_message_activity(self, turn_context: TurnContext):
        user_id = turn_context.activity.from_property.id or "anonymous"
        profile = self._users.get(user_id)

        text = (turn_context.activity.text or "").strip().lower()

        # 1. Allow the user to “tell” us their preference explicitly
        if "expert mode" in text:
            profile = self._users.update(user_id, pref_mode="expert", experience="expert")
            msg = MessageFactory.text(
                "Got it – switching you to **expert mode**. "
                "I’ll keep responses compact and show more options."
            )
            msg.attachments = [self._build_expert_card(profile)]
            await turn_context.send_activity(msg)
            return

        if "guided mode" in text or "beginner mode" in text:
            profile = self._users.update(user_id, pref_mode="guided", experience="new")
            msg = MessageFactory.text(
                "No problem – I’ll slow down and guide you step by step."
            )
            msg.attachments = [self._build_guided_card(profile)]
            await turn_context.send_activity(msg)
            return

        if "dark mode" in text:
            profile = self._users.update(user_id, theme="dark")
            await turn_context.send_activity(
                MessageFactory.text(
                    "Dark mode preference noted. I’ll style my cards accordingly."
                )
            )
            return

        if "light mode" in text:
            profile = self._users.update(user_id, theme="light")
            await turn_context.send_activity(
                MessageFactory.text(
                    "Switched back to light mode."
                )
            )
            return

        # 2. Simple example task: explain recommendations differently per user type
        if "beginner tips" in text or "tips" in text:
            profile = self._users.update(user_id, experience="intermediate")
            await turn_context.send_activity(
                MessageFactory.text(
                    "Here are a few tips:\n"
                    "1. Start with **/recommend 1 3** to see sample results.\n"
                    "2. Ask me to **explain recommender** if you want more detail.\n"
                    "3. When you feel comfortable, say **expert mode**."
                )
            )
            return

        if "explain recommender" in text:
            if profile.experience in ("new", "intermediate"):
                await turn_context.send_activity(
                    MessageFactory.text(
                        "I use a simple **hybrid recommender**:\n"
                        "• I look at items similar to what you interacted with.\n"
                        "• I mix that with content features (title, tags).\n"
                        "• If I know nothing about you, I fall back to popular items."
                    )
                )
            else:
                await turn_context.send_activity(
                    MessageFactory.text(
                        "Under the hood I compute item–item similarities from "
                        "implicit feedback and content TF-IDF, then blend them with "
                        "α=0.6. Cold users get popularity backfill."
                    )
                )
            return

        # 3. Default behaviour: adapt tone + card layout

        # Basic heuristic: if the user sends very short / ambiguous text,
        # count it as a potential “error” indicating confusion.
        if len(text) <= 1:
            profile.errors += 1

        # Choose card layout from preference
        if profile.pref_mode == "guided":
            card = self._build_guided_card(profile)
            response_text = (
                f"You said: **{text or '(no text)'}**.\n\n"
                "Because you’re in guided mode, I’ll keep the interface simple."
            )
        else:
            card = self._build_expert_card(profile)
            response_text = (
                f"You said: **{text or '(no text)'}**.\n\n"
                "You’re in expert mode, so I’ll show more shortcuts."
            )

        msg = MessageFactory.text(response_text)
        msg.attachments = [card]
        await turn_context.send_activity(msg)

    async def on_members_added_activity(
        self,
        members_added,
        turn_context: TurnContext,
    ):
        """
        Welcome message – describes the adaptive behaviour.
        """
        for member in members_added:
            if member.id != turn_context.activity.recipient.id:
                welcome = (
                    "Hi, I’m your **Adaptive UI** bot.\n\n"
                    "- Say **guided mode** if you want step-by-step help.\n"
                    "- Say **expert mode** if you want compact answers and more options.\n"
                    "- Say **dark mode** or **light mode** to change visual style."
                )
                await turn_context.send_activity(MessageFactory.text(welcome))
