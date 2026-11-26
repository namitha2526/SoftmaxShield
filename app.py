"""Streamlit entrypoint for SoftmaxShield UI."""
import importlib
import softmaxshield.ui.app as ui_app

# Always reload ui_app so any style or logic updates are picked up instantly without stale cache
importlib.reload(ui_app)

if __name__ == "__main__":
    ui_app.render_dashboard()
else:
    ui_app.render_dashboard()
