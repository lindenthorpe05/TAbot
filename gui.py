"""
gui.py - Chat interface for The Stable Boy agent.

Dark-themed conversational UI built with CustomTkinter.
Sends messages to the Ollama agent on a background thread
and displays responses with tool-execution status lines.
"""

import threading
import tkinter as tk

import customtkinter as ctk

from agent import Agent, list_models, DEFAULT_MODEL

# -- Appearance --------------------------------------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

NEON_GREEN = "#39FF14"
BG_BLACK = "#0A0A0A"
BG_DARK = "#141414"
BG_CARD = "#1A1A1A"
BG_ENTRY = "#1E1E1E"
BG_USER_MSG = "#1E1E1E"
BG_BOT_MSG = "#161616"
FG_WHITE = "#E0E0E0"
FG_DIM = "#808080"
FG_SUBTLE = "#555555"
BORDER_CLR = "#2A2A2A"
BTN_CLR = "#1E1E1E"
BTN_HOVER = "#2A2A2A"
AMBER = "#FFB300"

FONT_UI = ("Segoe UI", 11)
FONT_UI_SM = ("Segoe UI", 10)
FONT_UI_BOLD = ("Segoe UI", 11, "bold")
FONT_HEADER = ("Segoe UI", 13, "bold")
FONT_TITLE = ("Segoe UI", 20, "bold")
FONT_MONO = ("Consolas", 11)
FONT_MONO_SM = ("Consolas", 10)
FONT_MSG = ("Segoe UI", 12)
FONT_MSG_BOLD = ("Segoe UI", 12, "bold")

CORNER_R = 16
CORNER_R_SM = 12


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("THE STABLE BOY")
        self.geometry("900x700")
        self.minsize(600, 450)
        self.configure(fg_color=BG_BLACK)

        self._busy = False
        self._available_models = list_models()
        self._current_model = self._available_models[0] if self._available_models else DEFAULT_MODEL
        self._build_ui()
        self._init_agent()
        self._restore_chat()

    # ================================================================
    # Agent
    # ================================================================

    def _init_agent(self):
        self._agent = Agent(model=self._current_model, status_callback=self._on_status)

    def _restore_chat(self):
        """Replay saved conversation into the chat window, or show welcome."""
        has_history = False
        for msg in self._agent.history:
            role = msg.get("role") if isinstance(msg, dict) else None
            content = msg.get("content", "") if isinstance(msg, dict) else ""
            if role == "user":
                self._append_user(content)
                has_history = True
            elif role == "assistant" and content:
                self._append_bot(content)
                has_history = True

        if not has_history:
            self._append_bot(
                "G'day! I'm The Stable Boy. Tell me which race you want "
                "to analyse and I'll fire up the feeds, crunch the numbers, "
                "and find you some value. Just drop a TAB link or say "
                "something like \"scan Flemington race 4\"."
            )

    def _on_status(self, msg: str):
        """Called from agent (background thread) when tool status changes."""
        self.after(0, lambda: self._set_status(msg))
        if msg.startswith("Running "):
            tool = msg.removeprefix("Running ").rstrip(".")
            labels = {
                "scan_race": "Scanning TAB feeds...",
                "scan_next_races": "Scanning upcoming races...",
                "parse_data": "Parsing race data...",
                "predict_winners": "Running XGBoost model...",
            }
            status_text = labels.get(tool, msg)
            self.after(0, lambda t=status_text: self._append_status(t))

    # ================================================================
    # UI Construction
    # ================================================================

    def _build_ui(self):
        container = ctk.CTkFrame(self, fg_color=BG_BLACK, corner_radius=0)
        container.pack(fill="both", expand=True, padx=16, pady=12)

        # -- Top bar ----------------------------------------------------------
        top_bar = ctk.CTkFrame(
            container, fg_color=BG_DARK, corner_radius=CORNER_R,
            border_width=1, border_color=BORDER_CLR, height=56,
        )
        top_bar.pack(fill="x", pady=(0, 10))
        top_bar.pack_propagate(False)

        title_label = ctk.CTkLabel(
            top_bar, text="THE STABLE BOY", font=FONT_TITLE,
            text_color=NEON_GREEN,
        )
        title_label.pack(side="left", padx=20)

        # Model selector
        self._model_var = ctk.StringVar(value=self._current_model)
        self._model_dropdown = ctk.CTkOptionMenu(
            top_bar,
            values=self._available_models,
            variable=self._model_var,
            command=self._on_model_change,
            font=FONT_UI_SM,
            fg_color=BG_ENTRY, button_color=BTN_CLR,
            button_hover_color=BTN_HOVER,
            text_color=FG_WHITE,
            dropdown_fg_color=BG_DARK,
            dropdown_text_color=FG_WHITE,
            dropdown_hover_color=BTN_HOVER,
            corner_radius=CORNER_R_SM,
            height=32, width=200,
        )
        self._model_dropdown.pack(side="right", padx=(0, 10))

        model_label = ctk.CTkLabel(
            top_bar, text="MODEL", font=FONT_UI_SM,
            text_color=FG_SUBTLE,
        )
        model_label.pack(side="right", padx=(0, 4))

        # New Chat button
        new_chat_btn = ctk.CTkButton(
            top_bar, text="NEW CHAT", font=FONT_UI_BOLD,
            fg_color=BTN_CLR, hover_color=BTN_HOVER,
            text_color=FG_DIM, border_color=BORDER_CLR,
            border_width=1, corner_radius=CORNER_R_SM,
            height=32, width=100,
            command=self._on_new_chat,
        )
        new_chat_btn.pack(side="right", padx=(0, 20))

        # Status dot + text
        status_frame = ctk.CTkFrame(top_bar, fg_color="transparent")
        status_frame.pack(side="right", padx=20)

        self._status_dot = ctk.CTkLabel(
            status_frame, text="\u2B24", font=("Segoe UI", 8),
            text_color=NEON_GREEN,
        )
        self._status_dot.pack(side="left", padx=(0, 6))

        self._status_label = ctk.CTkLabel(
            status_frame, text="Ready", font=FONT_UI_SM,
            text_color=FG_DIM,
        )
        self._status_label.pack(side="left")

        # -- Chat area --------------------------------------------------------
        chat_card = ctk.CTkFrame(
            container, fg_color=BG_DARK, corner_radius=CORNER_R,
            border_width=1, border_color=BORDER_CLR,
        )
        chat_card.pack(fill="both", expand=True, pady=(0, 10))

        self._chat_box = ctk.CTkTextbox(
            chat_card, font=FONT_MSG, text_color=FG_WHITE,
            fg_color=BG_BLACK, corner_radius=CORNER_R_SM,
            wrap="word", activate_scrollbars=True,
            scrollbar_button_color=BORDER_CLR,
            scrollbar_button_hover_color=FG_DIM,
            state="disabled",
        )
        self._chat_box.pack(fill="both", expand=True, padx=10, pady=10)

        # Configure text tags for styling
        self._chat_box.tag_config("user_name", foreground=FG_DIM)
        self._chat_box.tag_config("bot_name", foreground=NEON_GREEN)
        self._chat_box.tag_config("user_msg", foreground=FG_WHITE)
        self._chat_box.tag_config("bot_msg", foreground=FG_WHITE)
        self._chat_box.tag_config("status_msg", foreground=FG_SUBTLE)
        self._chat_box.tag_config("spacer")

        # -- Input bar --------------------------------------------------------
        input_card = ctk.CTkFrame(
            container, fg_color=BG_DARK, corner_radius=CORNER_R,
            border_width=1, border_color=BORDER_CLR, height=60,
        )
        input_card.pack(fill="x")
        input_card.pack_propagate(False)

        input_inner = ctk.CTkFrame(input_card, fg_color="transparent")
        input_inner.pack(fill="both", expand=True, padx=14, pady=10)

        self._input = ctk.CTkEntry(
            input_inner,
            placeholder_text="Type a message...",
            font=FONT_MSG, text_color=FG_WHITE,
            fg_color=BG_ENTRY, border_color=BORDER_CLR,
            corner_radius=CORNER_R_SM, height=40,
        )
        self._input.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self._input.bind("<Return>", lambda e: self._on_send())

        self._send_btn = ctk.CTkButton(
            input_inner, text="SEND", font=FONT_HEADER,
            fg_color=BTN_CLR, hover_color=BTN_HOVER,
            text_color=NEON_GREEN, border_color=NEON_GREEN,
            border_width=1, corner_radius=CORNER_R_SM,
            height=40, width=90,
            command=self._on_send,
        )
        self._send_btn.pack(side="right")

    def _on_model_change(self, model_name: str):
        """Switch the agent to a different model."""
        if self._busy:
            self._model_var.set(self._current_model)
            return
        self._current_model = model_name
        self._agent.model = model_name
        self._append_status(f"Switched model to {model_name}")

    def _on_new_chat(self):
        """Clear history and reset the chat window."""
        if self._busy:
            return
        self._agent.clear_history()
        self._chat_box.configure(state="normal")
        self._chat_box.delete("1.0", "end")
        self._chat_box.configure(state="disabled")
        self._append_bot(
            "G'day! I'm The Stable Boy. Tell me which race you want "
            "to analyse and I'll fire up the feeds, crunch the numbers, "
            "and find you some value. Just drop a TAB link or say "
            "something like \"scan Flemington race 4\"."
        )

    # ================================================================
    # Chat display helpers
    # ================================================================

    def _append_text(self, text: str, tag: str):
        self._chat_box.configure(state="normal")
        self._chat_box.insert("end", text, tag)
        self._chat_box.see("end")
        self._chat_box.configure(state="disabled")

    def _append_user(self, text: str):
        self._append_text("You\n", "user_name")
        self._append_text(text + "\n\n", "user_msg")

    def _append_bot(self, text: str):
        self._append_text("The Stable Boy\n", "bot_name")
        self._append_text(text + "\n\n", "bot_msg")

    def _append_status(self, text: str):
        self._append_text(f"  \u2022 {text}\n", "status_msg")

    # ================================================================
    # Status bar
    # ================================================================

    def _set_status(self, text: str):
        self._status_label.configure(text=text)
        if text == "Ready":
            self._status_dot.configure(text_color=NEON_GREEN)
        else:
            self._status_dot.configure(text_color=AMBER)

    # ================================================================
    # Send / Receive
    # ================================================================

    def _on_send(self):
        if self._busy:
            return

        text = self._input.get().strip()
        if not text:
            return

        self._input.delete(0, "end")
        self._append_user(text)

        self._busy = True
        self._send_btn.configure(state="disabled")
        self._set_status("Thinking...")

        thread = threading.Thread(target=self._run_agent, args=(text,), daemon=True)
        thread.start()

    def _run_agent(self, user_text: str):
        try:
            reply = self._agent.chat(user_text)
        except Exception as e:
            reply = f"Something went wrong: {e}"

        self.after(0, lambda: self._receive_reply(reply))

    def _receive_reply(self, text: str):
        self._append_bot(text)
        self._busy = False
        self._send_btn.configure(state="normal")
        self._set_status("Ready")


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
