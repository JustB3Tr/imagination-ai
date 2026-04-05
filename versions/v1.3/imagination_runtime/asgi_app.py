"""
FastAPI + Gradio: email signup/login cookies, optional Google OAuth, shared logout.

Optional Next.js static UI (single process with the model): build the frontend with
IMAGINATION_NEXT_EXPORT=1 and IMAGINATION_NEXT_BASE_PATH=/next, then run python app.py.
The export is served at /next/ alongside Gradio at /. Set IMAGINATION_NEXT_UI=0 to skip.

Do not use `from __future__ import annotations` in this module: route handlers are
defined inside build_full_app(), and postponed annotations can prevent FastAPI from
recognizing Starlette Request. It then treats a parameter named "request" as a required
query field (422 missing request on /auth/me and /auth/email/login).
"""
import os
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

from starlette.requests import Request


def _next_static_root() -> Optional[Path]:
    off = (os.getenv("IMAGINATION_NEXT_UI") or "").strip().lower()
    if off in ("0", "false", "no", "off"):
        return None
    raw = (os.getenv("IMAGINATION_NEXT_EXPORT_DIR") or "").strip()
    if raw:
        p = Path(raw).expanduser().resolve()
        return p if p.is_dir() and (p / "index.html").is_file() else None
    here = Path(__file__).resolve().parent.parent / "front&back" / "frontend" / "out"
    return here if here.is_dir() and (here / "index.html").is_file() else None


def _session_cookie_kwargs(request: Any) -> dict:
    """Browsers on HTTPS (e.g. gradio.live tunnel) need Secure cookies."""
    secure = getattr(getattr(request, "url", None), "scheme", None) == "https"
    return {
        "httponly": True,
        "max_age": 86400 * 30,
        "samesite": "lax",
        "secure": secure,
        "path": "/",
    }


def build_full_app(demo: Any) -> Any:
    import gradio as gr
    from fastapi import FastAPI, Form
    from fastapi.responses import JSONResponse, RedirectResponse
    from urllib.parse import parse_qs

    app = FastAPI()

    @app.get("/auth/me")
    async def auth_me(http_request: Request):
        """
        Read session cookie on a normal GET (Gradio's load event often has no Cookie header).
        The UI fetches this from the browser so sign-in persists after redirect.
        """
        from imagination_runtime.paths import resolve_root_path
        from imagination_runtime.session_signed import user_id_from_cookie
        from imagination_runtime.users import (
            get_user_by_id,
            get_user_chat_thread,
            load_learner_profile,
            load_user_memory,
            pick_default_thread_id,
        )

        root = resolve_root_path(None)
        extra = ""
        raw_q = http_request.url.query or ""
        if raw_q:
            pq = parse_qs(str(raw_q))
            if (pq.get("login") or [""])[0] == "failed":
                extra = "\n\n*Sign-in failed — check email and password.*"
            su = (pq.get("signup_error") or [""])[0]
            if su:
                extra += f"\n\n*Sign-up:* {su}"

        uid = user_id_from_cookie(http_request.cookies.get("imagination_uid"))
        if not uid:
            return JSONResponse({"logged_in": False, "banner_extra": extra})
        u = get_user_by_id(root, uid)
        if not u:
            return JSONResponse({"logged_in": False, "banner_extra": extra})
        tid = pick_default_thread_id(root, uid)
        conv: list = []
        disp: list = []
        if tid:
            row = get_user_chat_thread(root, uid, tid)
            if row:
                conv = row.get("conversation") or []
                disp = row.get("display") or []
        notes = load_user_memory(root, uid)
        pa, pg, ps = load_learner_profile(root, uid)
        return JSONResponse(
            {
                "logged_in": True,
                "id": u.id,
                "name": u.display_name or u.email or "User",
                "email": u.email or "",
                "conv": conv,
                "disp": disp,
                "notes": notes,
                "profile_about": pa,
                "profile_goals": pg,
                "profile_skills": ps,
                "banner_extra": extra,
            }
        )

    @app.post("/auth/email/login")
    async def email_login(
        http_request: Request,
        email: str = Form(...),
        password: str = Form(...),
    ):
        from imagination_runtime.auth import login_email_password
        from imagination_runtime.paths import resolve_root_path
        from imagination_runtime.session_signed import sign_user_id

        root = resolve_root_path(None)
        u = login_email_password(root, email, password)
        if not u:
            return RedirectResponse(url="/?login=failed", status_code=302)
        resp = RedirectResponse(url="/", status_code=302)
        ck = _session_cookie_kwargs(http_request)
        resp.set_cookie("imagination_uid", sign_user_id(u.id), **ck)
        return resp

    @app.post("/auth/email/signup")
    async def email_signup(
        http_request: Request,
        email: str = Form(...),
        password: str = Form(...),
        display_name: str = Form(""),
    ):
        from imagination_runtime.auth import signup_email_password
        from imagination_runtime.paths import resolve_root_path
        from imagination_runtime.session_signed import sign_user_id

        root = resolve_root_path(None)
        user, err = signup_email_password(root, email, password, display_name or "")
        if err or not user:
            return RedirectResponse(url=f"/?signup_error={quote(err or 'signup failed')}", status_code=302)
        resp = RedirectResponse(url="/", status_code=302)
        ck = _session_cookie_kwargs(http_request)
        resp.set_cookie("imagination_uid", sign_user_id(user.id), **ck)
        return resp

    @app.get("/auth/logout")
    async def logout(http_request: Request):
        resp = RedirectResponse(url="/", status_code=302)
        ck = _session_cookie_kwargs(http_request)
        resp.delete_cookie("imagination_uid", path="/", secure=ck["secure"], httponly=True, samesite="lax")
        return resp

    if os.getenv("GOOGLE_CLIENT_ID") and os.getenv("GOOGLE_CLIENT_SECRET"):
        try:
            from authlib.integrations.starlette_client import OAuth
            from starlette.middleware.sessions import SessionMiddleware

            secret = os.environ.get("IMAGINATION_SESSION_SECRET", "change-me-in-production")
            app.add_middleware(SessionMiddleware, secret_key=secret)
            oauth = OAuth()
            oauth.register(
                name="google",
                client_id=os.environ["GOOGLE_CLIENT_ID"],
                client_secret=os.environ["GOOGLE_CLIENT_SECRET"],
                server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
                client_kwargs={"scope": "openid email profile"},
            )

            @app.get("/auth/google/login")
            async def google_login(http_request: Request):
                redir = os.environ.get("GOOGLE_REDIRECT_URI")
                if not redir:
                    redir = str(http_request.base_url).rstrip("/") + "/auth/google/callback"
                return await oauth.google.authorize_redirect(http_request, redir)

            @app.get("/auth/google/callback")
            async def google_callback(http_request: Request):
                from imagination_runtime.paths import resolve_root_path
                from imagination_runtime.session_signed import sign_user_id
                from imagination_runtime.users import get_or_create_user

                token = await oauth.google.authorize_access_token(http_request)
                userinfo = token.get("userinfo")
                if userinfo is None:
                    try:
                        userinfo = await oauth.google.userinfo(token=token)
                    except Exception:
                        userinfo = {}
                if not userinfo:
                    return RedirectResponse(url="/?login=failed", status_code=302)
                sub = userinfo.get("sub")
                email = userinfo.get("email") or ""
                name = userinfo.get("name") or (email.split("@")[0] if email else "User")
                if not sub:
                    return RedirectResponse(url="/?login=failed", status_code=302)
                root = resolve_root_path(None)
                u = get_or_create_user(
                    root,
                    provider="google",
                    provider_uid=str(sub),
                    display_name=name,
                    email=email,
                )
                resp = RedirectResponse(url="/", status_code=302)
                ck = _session_cookie_kwargs(http_request)
                resp.set_cookie("imagination_uid", sign_user_id(u.id), **ck)
                return resp
        except ImportError:
            pass

    @app.get("/api/health")
    async def api_health():
        return JSONResponse({"ok": True, "service": "imagination-v1.3"})

    @app.get("/api/info")
    async def api_info():
        nxt = _next_static_root()
        return JSONResponse(
            {
                "version": "v1.3",
                "gradio_path": "/",
                "next_static_path": "/next/" if nxt else None,
                "next_static_dir": str(nxt) if nxt else None,
            }
        )

    next_root = _next_static_root()
    if next_root is not None:
        from starlette.staticfiles import StaticFiles

        app.mount(
            "/next",
            StaticFiles(directory=str(next_root), html=True),
            name="next_static",
        )
        print(
            f"[imagination] Next static UI → http://127.0.0.1:<port>/next/ (from {next_root})",
            flush=True,
        )

    return gr.mount_gradio_app(app, demo, path="/")
