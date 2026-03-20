"""Context engine: detects topics, generates AI responses, and produces panel content."""

from __future__ import annotations

import json
import os
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Context catalogue
# ---------------------------------------------------------------------------

CONTEXT_DEFS: Dict[str, Dict[str, Any]] = {
    "code": {
        "label": "Code",
        "icon": "\U0001f4bb",
        "color": "#8ea2ff",
        "keywords": [
            "code", "function", "class", "method", "bug", "error", "python",
            "javascript", "typescript", "rust", "java", "c++", "program",
            "debug", "fix", "compile", "script", "algorithm", "loop",
            "variable", "array", "list", "import", "return", "syntax", "api",
            "endpoint", "implement", "html", "css", "react", "vue", "node",
            "flask", "django", "database", "sql", "query", "refactor",
            "recursion", "sort", "search", "binary", "linked list", "tree",
            "stack", "queue", "hash", "regex", "parse", "decorator",
            "lambda", "async", "await", "promise", "callback",
        ],
        "patterns": [
            r"```", r"\bdef\s+\w+", r"\bclass\s+\w+", r"\bfunction\s+\w+",
            r"\bimport\s+\w+", r"\bfrom\s+\w+\s+import", r"\bconst\s+\w+",
            r"\blet\s+\w+", r"\bvar\s+\w+", r"\bprint\s*\(",
        ],
    },
    "data": {
        "label": "Data",
        "icon": "\U0001f4ca",
        "color": "#66e0ff",
        "keywords": [
            "data", "table", "csv", "spreadsheet", "dataset", "statistics",
            "column", "row", "excel", "dataframe", "pandas", "analysis",
            "average", "mean", "median", "count", "sum", "aggregate",
            "filter", "group by", "join", "merge", "pivot", "sample",
        ],
        "patterns": [r"\d+\s*[,|]\s*\d+\s*[,|]\s*\d+"],
    },
    "terminal": {
        "label": "Terminal",
        "icon": "\u2328\ufe0f",
        "color": "#7efcc3",
        "keywords": [
            "terminal", "command", "shell", "bash", "cmd", "powershell",
            "run", "execute", "install", "npm", "pip", "apt", "brew",
            "docker", "kubernetes", "deploy", "ssh", "git", "chmod",
            "mkdir", "grep", "curl", "wget", "sudo", "server", "port",
            "environment", "config", "systemctl", "service",
        ],
        "patterns": [
            r"\$\s+\w+", r"\bsudo\s+", r"\bnpm\s+(run|install|start)",
            r"\bpip\s+install", r"\bdocker\s+\w+", r"\bgit\s+\w+",
        ],
    },
    "preview": {
        "label": "Preview",
        "icon": "\U0001f4dd",
        "color": "#ffd27a",
        "keywords": [
            "documentation", "document", "readme", "markdown", "article",
            "blog", "essay", "summarize", "explain", "describe", "outline",
            "draft", "report", "proposal", "letter", "email", "content",
            "write a", "math", "equation", "formula", "calculate",
            "integral", "derivative", "matrix", "algebra", "geometry",
            "probability", "theorem", "proof",
        ],
        "patterns": [],
    },
    "chart": {
        "label": "Chart",
        "icon": "\U0001f4c8",
        "color": "#c08aff",
        "keywords": [
            "chart", "graph", "plot", "visualization", "bar chart",
            "line chart", "pie chart", "scatter", "histogram", "trend",
            "matplotlib", "plotly", "seaborn", "visualize", "dashboard",
        ],
        "patterns": [r"\bplt\.\w+", r"\bfig\.\w+"],
    },
}

# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class PanelPayload:
    visible: bool = False
    content: Any = None
    language: str = "python"
    title: str = ""


@dataclass
class AIResponse:
    text: str = ""
    panels: Dict[str, PanelPayload] = field(default_factory=dict)
    detected_contexts: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Context detection
# ---------------------------------------------------------------------------

def detect_contexts(
    message: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> List[Tuple[str, float]]:
    """Return ``[(context_name, confidence)]`` sorted by confidence descending."""
    msg_lower = message.lower()
    scores: Dict[str, float] = {}

    for ctx, info in CONTEXT_DEFS.items():
        score = 0.0
        for kw in info["keywords"]:
            if kw in msg_lower:
                score += 1.0
        for pat in info["patterns"]:
            if re.search(pat, message, re.IGNORECASE):
                score += 1.5
        if history:
            for m in (history or [])[-4:]:
                c = (m.get("content") or "").lower()
                for kw in info["keywords"][:8]:
                    if kw in c:
                        score += 0.25
        if score > 0:
            scores[ctx] = min(score / 3.0, 1.0)

    return sorted(scores.items(), key=lambda x: -x[1])


def badges_html(contexts: List[Tuple[str, float]]) -> str:
    if not contexts:
        return ""
    parts: List[str] = []
    for name, conf in contexts:
        info = CONTEXT_DEFS.get(name, {})
        color = info.get("color", "#888")
        label = info.get("label", name)
        icon = info.get("icon", "")
        opacity = 0.5 + conf * 0.5
        parts.append(
            f"<span style='display:inline-flex;align-items:center;gap:4px;"
            f"padding:4px 12px;border-radius:999px;font-size:12px;font-weight:600;"
            f"background:{color}22;color:{color};border:1px solid {color}44;"
            f"opacity:{opacity:.2f};letter-spacing:.4px;'>"
            f"{icon} {label}</span>"
        )
    return (
        "<div style='display:flex;flex-wrap:wrap;gap:8px;"
        "margin-bottom:4px;'>" + "".join(parts) + "</div>"
    )


# ---------------------------------------------------------------------------
# Language detection helper
# ---------------------------------------------------------------------------

_LANG_KEYWORDS = {
    "python": ["python", "py", "django", "flask", "pandas", "numpy", "pip"],
    "javascript": ["javascript", "js", "node", "react", "vue", "angular", "npm"],
    "typescript": ["typescript", "ts", "tsx"],
    "rust": ["rust", "cargo", "fn "],
    "java": ["java", "spring", "maven", "gradle"],
    "go": ["golang", "go", "goroutine"],
    "bash": ["bash", "shell", "sh ", "zsh"],
    "sql": ["sql", "select", "insert", "update", "delete", "create table"],
    "html": ["html", "dom", "element"],
    "css": ["css", "style", "flexbox", "grid"],
}


def detect_language(message: str) -> str:
    msg = message.lower()
    best, best_score = "python", 0
    for lang, kws in _LANG_KEYWORDS.items():
        score = sum(1 for k in kws if k in msg)
        if score > best_score:
            best, best_score = lang, score
    return best


# ---------------------------------------------------------------------------
# Code generation (built-in templates)
# ---------------------------------------------------------------------------

_CODE_BANK: Dict[str, Dict[str, str]] = {
    "sort": {
        "python": textwrap.dedent("""\
            def sort_list(items, *, reverse=False):
                \"\"\"Return a sorted copy of *items*.\"\"\"
                return sorted(items, reverse=reverse)

            numbers = [64, 34, 25, 12, 22, 11, 90]
            print("Original:", numbers)
            print("Sorted:  ", sort_list(numbers))
            print("Reversed:", sort_list(numbers, reverse=True))
        """),
        "javascript": textwrap.dedent("""\
            function sortList(items, reverse = false) {
              return [...items].sort((a, b) => reverse ? b - a : a - b);
            }

            const nums = [64, 34, 25, 12, 22, 11, 90];
            console.log("Sorted:", sortList(nums));
        """),
    },
    "fibonacci": {
        "python": textwrap.dedent("""\
            def fibonacci(n):
                \"\"\"Return the first *n* Fibonacci numbers.\"\"\"
                a, b, out = 0, 1, []
                for _ in range(n):
                    out.append(a)
                    a, b = b, a + b
                return out

            print(fibonacci(12))
        """),
    },
    "factorial": {
        "python": textwrap.dedent("""\
            def factorial(n):
                \"\"\"Compute n! iteratively.\"\"\"
                result = 1
                for i in range(2, n + 1):
                    result *= i
                return result

            for x in range(8):
                print(f"{x}! = {factorial(x)}")
        """),
    },
    "palindrome": {
        "python": textwrap.dedent("""\
            def is_palindrome(text):
                \"\"\"Check whether *text* is a palindrome (case-insensitive).\"\"\"
                cleaned = ''.join(ch.lower() for ch in text if ch.isalnum())
                return cleaned == cleaned[::-1]

            tests = ["racecar", "hello", "A man a plan a canal Panama"]
            for t in tests:
                print(f"{t!r:>40s}  ->  {is_palindrome(t)}")
        """),
    },
    "api": {
        "python": textwrap.dedent("""\
            from flask import Flask, jsonify, request

            app = Flask(__name__)
            items = []

            @app.route("/items", methods=["GET"])
            def list_items():
                return jsonify(items)

            @app.route("/items", methods=["POST"])
            def create_item():
                data = request.get_json(force=True)
                items.append(data)
                return jsonify(data), 201

            if __name__ == "__main__":
                app.run(debug=True)
        """),
    },
    "scrape": {
        "python": textwrap.dedent("""\
            import requests
            from bs4 import BeautifulSoup

            def scrape_titles(url):
                \"\"\"Return all <h2> titles from *url*.\"\"\"
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                return [h.get_text(strip=True) for h in soup.find_all("h2")]

            titles = scrape_titles("https://example.com")
            for t in titles:
                print("-", t)
        """),
    },
    "todo": {
        "python": textwrap.dedent("""\
            class TodoList:
                def __init__(self):
                    self._items = []

                def add(self, task):
                    self._items.append({"task": task, "done": False})

                def complete(self, index):
                    self._items[index]["done"] = True

                def show(self):
                    for i, item in enumerate(self._items):
                        mark = "x" if item["done"] else " "
                        print(f"  [{mark}] {i}. {item['task']}")

            todos = TodoList()
            todos.add("Buy groceries")
            todos.add("Write unit tests")
            todos.complete(0)
            todos.show()
        """),
    },
    "calculator": {
        "python": textwrap.dedent("""\
            def calculate(expression):
                \"\"\"Evaluate a simple arithmetic expression safely.\"\"\"
                allowed = set("0123456789+-*/(). ")
                if not all(ch in allowed for ch in expression):
                    raise ValueError("Invalid characters in expression")
                return eval(expression)  # safe for digits & operators only

            tests = ["2 + 3 * 4", "(10 - 3) * 2", "100 / 4 + 1"]
            for expr in tests:
                print(f"  {expr} = {calculate(expr)}")
        """),
    },
    "binary_search": {
        "python": textwrap.dedent("""\
            def binary_search(arr, target):
                lo, hi = 0, len(arr) - 1
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if arr[mid] == target:
                        return mid
                    elif arr[mid] < target:
                        lo = mid + 1
                    else:
                        hi = mid - 1
                return -1

            data = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]
            for val in [23, 50]:
                idx = binary_search(data, val)
                print(f"Search {val}: index={idx}")
        """),
    },
}

_GENERIC_CODE = textwrap.dedent("""\
    def solve(input_data):
        \"\"\"Process the input and return a result.\"\"\"
        result = []
        for item in input_data:
            result.append(item)
        return result

    print(solve([1, 2, 3]))
""")


def generate_code(message: str) -> Tuple[str, str]:
    """Return ``(code_string, language)``."""
    msg = message.lower()
    lang = detect_language(message)
    for key, templates in _CODE_BANK.items():
        if key in msg:
            code = templates.get(lang, templates.get("python", _GENERIC_CODE))
            return code.strip(), lang
    return _GENERIC_CODE.strip(), lang


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

_DATA_BANK: Dict[str, Dict[str, Any]] = {
    "sales": {
        "headers": ["Product", "Q1", "Q2", "Q3", "Q4", "Total"],
        "rows": [
            ["Widget A", 12400, 15800, 18200, 21000, 67400],
            ["Widget B", 8200, 9500, 11400, 13100, 42200],
            ["Widget C", 20100, 22300, 19800, 25400, 87600],
            ["Gadget X", 5600, 7800, 9100, 11200, 33700],
            ["Gadget Y", 14300, 16700, 15200, 18900, 65100],
        ],
    },
    "students": {
        "headers": ["Name", "Math", "Science", "English", "GPA"],
        "rows": [
            ["Alice", 92, 88, 95, 3.8],
            ["Bob", 78, 85, 82, 3.2],
            ["Carol", 95, 97, 91, 3.9],
            ["Dave", 61, 72, 68, 2.5],
            ["Eve", 88, 90, 87, 3.6],
        ],
    },
    "weather": {
        "headers": ["City", "High (F)", "Low (F)", "Humidity %", "Wind (mph)"],
        "rows": [
            ["New York", 72, 58, 65, 12],
            ["London", 61, 50, 78, 8],
            ["Tokyo", 77, 64, 70, 6],
            ["Sydney", 68, 55, 60, 15],
            ["Paris", 64, 52, 72, 10],
        ],
    },
    "employees": {
        "headers": ["Name", "Department", "Role", "Experience (yr)", "Salary ($)"],
        "rows": [
            ["Kim", "Engineering", "Senior Dev", 8, 145000],
            ["Jay", "Design", "UX Lead", 6, 125000],
            ["Lee", "Marketing", "Manager", 10, 118000],
            ["Mia", "Engineering", "Staff Dev", 12, 175000],
            ["Sam", "Data", "Analyst", 3, 95000],
        ],
    },
}

_GENERIC_DATA = {
    "headers": ["ID", "Category", "Value", "Status"],
    "rows": [
        [1, "Alpha", 42, "Active"],
        [2, "Beta", 87, "Active"],
        [3, "Gamma", 15, "Inactive"],
        [4, "Delta", 63, "Active"],
        [5, "Epsilon", 29, "Pending"],
    ],
}


def generate_data(message: str) -> Dict[str, Any]:
    msg = message.lower()
    for key, dataset in _DATA_BANK.items():
        if key in msg:
            return dataset
    return _GENERIC_DATA


# ---------------------------------------------------------------------------
# Terminal generation
# ---------------------------------------------------------------------------

_TERM_BANK: Dict[str, str] = {
    "docker": textwrap.dedent("""\
        # Build and run a Docker container
        $ docker build -t myapp:latest .
        $ docker run -d -p 8080:8080 --name myapp myapp:latest
        $ docker ps
        CONTAINER ID  IMAGE         STATUS        PORTS
        a1b2c3d4e5f6  myapp:latest  Up 10 sec     0.0.0.0:8080->8080/tcp

        # View logs
        $ docker logs -f myapp
    """),
    "git": textwrap.dedent("""\
        # Common git workflow
        $ git status
        $ git add -A
        $ git commit -m "feat: add new feature"
        $ git push origin main

        # Branching
        $ git checkout -b feature/new-ui
        $ git merge main
    """),
    "pip": textwrap.dedent("""\
        # Create a virtual environment and install deps
        $ python -m venv .venv
        $ source .venv/bin/activate
        $ pip install -r requirements.txt
        $ pip list
    """),
    "npm": textwrap.dedent("""\
        # Initialize and run a Node.js project
        $ npm init -y
        $ npm install express cors dotenv
        $ npm run dev

        > myapp@1.0.0 dev
        > node server.js
        Server running on http://localhost:3000
    """),
    "deploy": textwrap.dedent("""\
        # Deploy with Docker Compose
        $ docker compose build
        $ docker compose up -d
        $ docker compose ps
        NAME       SERVICE   STATUS    PORTS
        web        web       running   0.0.0.0:80->80/tcp
        db         postgres  running   5432/tcp
        cache      redis     running   6379/tcp
    """),
    "setup": textwrap.dedent("""\
        # Project setup
        $ mkdir myproject && cd myproject
        $ git init
        $ python -m venv .venv
        $ source .venv/bin/activate
        $ pip install flask gunicorn
        $ touch app.py requirements.txt
        $ echo "flask\\ngunicorn" > requirements.txt
    """),
}

_GENERIC_TERM = textwrap.dedent("""\
    $ echo "Hello from the terminal!"
    Hello from the terminal!

    $ python --version
    Python 3.11.7

    $ which python
    /usr/bin/python
""")


def generate_terminal(message: str) -> str:
    msg = message.lower()
    for key, content in _TERM_BANK.items():
        if key in msg:
            return content.strip()
    return _GENERIC_TERM.strip()


# ---------------------------------------------------------------------------
# Chart generation (requires matplotlib — graceful degradation)
# ---------------------------------------------------------------------------

def _try_generate_chart(message: str):
    """Return a matplotlib Figure or ``None`` if matplotlib is unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    msg = message.lower()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    fig.patch.set_facecolor("#0b1020")
    ax.set_facecolor("#0b1020")
    ax.tick_params(colors="#b0b8d0")
    ax.spines["bottom"].set_color("#333")
    ax.spines["left"].set_color("#333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.label.set_color("#b0b8d0")
    ax.yaxis.label.set_color("#b0b8d0")
    ax.title.set_color("#e0e4f0")

    palette = ["#8ea2ff", "#66e0ff", "#7efcc3", "#ff8ea2", "#ffd27a", "#c08aff"]

    if "pie" in msg:
        labels = ["Engineering", "Design", "Marketing", "Sales", "Support"]
        sizes = [35, 20, 18, 15, 12]
        ax.pie(sizes, labels=labels, colors=palette, autopct="%1.0f%%",
               textprops={"color": "#e0e4f0", "fontsize": 10})
        ax.set_title("Department Distribution")
    elif "line" in msg or "trend" in msg:
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        values = [12, 19, 15, 28, 35, 40, 38, 45, 52, 48, 61, 70]
        ax.plot(months, values, color=palette[0], linewidth=2.4, marker="o",
                markersize=5, markerfacecolor=palette[1])
        ax.fill_between(months, values, alpha=0.08, color=palette[0])
        ax.set_title("Monthly Growth")
        ax.set_ylabel("Users (k)")
    elif "scatter" in msg:
        import random
        random.seed(42)
        x = [random.gauss(50, 15) for _ in range(60)]
        y = [xi * 0.6 + random.gauss(0, 8) for xi in x]
        ax.scatter(x, y, c=palette[0], alpha=0.65, edgecolors=palette[1], linewidths=0.5)
        ax.set_title("Correlation Plot")
        ax.set_xlabel("Feature A")
        ax.set_ylabel("Feature B")
    elif "histogram" in msg:
        import random
        random.seed(42)
        data = [random.gauss(70, 12) for _ in range(200)]
        ax.hist(data, bins=20, color=palette[0], edgecolor=palette[1], alpha=0.75)
        ax.set_title("Score Distribution")
        ax.set_xlabel("Score")
        ax.set_ylabel("Frequency")
    else:
        categories = ["Product A", "Product B", "Product C", "Product D", "Product E"]
        values = [23, 45, 56, 78, 32]
        bars = ax.bar(categories, values, color=palette[:5], edgecolor="#1a1f30", linewidth=0.5)
        ax.set_title("Sales by Product")
        ax.set_ylabel("Revenue ($k)")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    str(val), ha="center", va="bottom", color="#b0b8d0", fontsize=9)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Preview / docs generation
# ---------------------------------------------------------------------------

_PREVIEW_TEMPLATES: Dict[str, str] = {
    "readme": textwrap.dedent("""\
        # My Project

        > A brief tagline describing the project.

        ## Getting Started

        ```bash
        git clone https://github.com/user/project.git
        cd project
        pip install -r requirements.txt
        python app.py
        ```

        ## Features
        - **Fast** — optimized for speed
        - **Simple** — minimal configuration
        - **Extensible** — plugin architecture

        ## License
        MIT
    """),
    "math": textwrap.dedent("""\
        ## Mathematical Reference

        **Quadratic Formula**

        $$x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$

        **Euler's Identity**

        $$e^{i\\pi} + 1 = 0$$

        **Pythagorean Theorem**

        $$a^2 + b^2 = c^2$$

        **Derivative Rules**

        | Rule | Formula |
        |------|---------|
        | Power | $\\frac{d}{dx} x^n = nx^{n-1}$ |
        | Chain | $\\frac{d}{dx} f(g(x)) = f'(g(x)) \\cdot g'(x)$ |
        | Product | $(fg)' = f'g + fg'$ |
    """),
    "blog": textwrap.dedent("""\
        # The Rise of Contextual User Interfaces

        Modern applications are moving beyond static layouts toward interfaces
        that **adapt in real time** based on what the user is doing.

        ## Why Context Matters

        Traditional UIs present every possible action at once, overwhelming users
        with irrelevant controls. A contextual UI surfaces only what's needed
        *right now*, reducing cognitive load.

        ## Key Principles

        1. **Detect** — understand the user's current task
        2. **Adapt** — show relevant tools and hide distractions
        3. **Transition** — change smoothly to avoid disorientation
        4. **Remember** — learn from interaction patterns

        ---

        *Contextual UI is not about removing features — it's about showing the
        right feature at the right time.*
    """),
}


def generate_preview(message: str) -> str:
    msg = message.lower()
    for key, content in _PREVIEW_TEMPLATES.items():
        if key in msg:
            return content.strip()
    if any(w in msg for w in ["equation", "formula", "math", "calculate"]):
        return _PREVIEW_TEMPLATES["math"].strip()
    return _PREVIEW_TEMPLATES["blog"].strip()


# ---------------------------------------------------------------------------
# Built-in AI backend
# ---------------------------------------------------------------------------

class BuiltInBackend:
    """Rule-based backend that works without any API keys."""

    def respond(
        self,
        message: str,
        history: List[Dict[str, str]],
        contexts: List[Tuple[str, float]],
    ) -> AIResponse:
        ctx_names = [c for c, _ in contexts]
        panels: Dict[str, PanelPayload] = {}
        text_parts: List[str] = []

        if "code" in ctx_names:
            code, lang = generate_code(message)
            panels["code"] = PanelPayload(visible=True, content=code, language=lang)
            text_parts.append(
                "I've generated the code for you — check the **Code** panel on the right. "
                "Feel free to ask me to modify or extend it."
            )

        if "data" in ctx_names:
            dataset = generate_data(message)
            panels["data"] = PanelPayload(visible=True, content=dataset)
            text_parts.append(
                "Here's the data in the **Data** panel. "
                "You can sort the columns by clicking the headers."
            )

        if "chart" in ctx_names:
            fig = _try_generate_chart(message)
            if fig is not None:
                panels["chart"] = PanelPayload(visible=True, content=fig)
                text_parts.append(
                    "I've created a visualization in the **Chart** panel."
                )

        if "terminal" in ctx_names:
            cmds = generate_terminal(message)
            panels["terminal"] = PanelPayload(visible=True, content=cmds, language="shell")
            text_parts.append(
                "The relevant commands are in the **Terminal** panel."
            )

        if "preview" in ctx_names and "code" not in ctx_names:
            md = generate_preview(message)
            panels["preview"] = PanelPayload(visible=True, content=md)
            text_parts.append(
                "I've prepared the content in the **Preview** panel."
            )

        if not text_parts:
            text_parts.append(
                "I'm here to help! Try asking me to:\n\n"
                "- **Write code** — *\"write a Python function to sort a list\"*\n"
                "- **Show data** — *\"show me some sales data\"*\n"
                "- **Create a chart** — *\"make a bar chart of product sales\"*\n"
                "- **Run commands** — *\"how do I set up Docker?\"*\n"
                "- **Draft docs** — *\"write a README for my project\"*\n\n"
                "The UI panels will adapt to whatever you're working on!"
            )

        return AIResponse(
            text="\n\n".join(text_parts),
            panels=panels,
            detected_contexts=ctx_names,
        )


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------

_OPENAI_SYSTEM = textwrap.dedent("""\
    You are an AI assistant powering a **contextual UI** — the interface around
    you changes dynamically based on the conversation.

    When you respond, return **valid JSON** with this schema:

    {
      "response": "<your markdown response to the user>",
      "panels": {
        "code":     {"visible": bool, "content": "<source code>", "language": "<lang>"},
        "data":     {"visible": bool, "content": {"headers": [...], "rows": [[...], ...]}},
        "terminal": {"visible": bool, "content": "<shell commands / output>"},
        "preview":  {"visible": bool, "content": "<markdown text>"},
        "chart":    {"visible": bool, "content": "<short description for chart type: bar|line|pie|scatter|histogram>"}
      }
    }

    Rules:
    - Only include panels that are relevant. Omit or set visible=false for irrelevant ones.
    - For code, generate complete, runnable snippets with examples.
    - For data, provide realistic sample data.
    - For terminal, provide commands the user can copy-paste.
    - For preview, provide well-formatted markdown.
    - For chart, describe the chart type and data — the frontend renders it.
    - Keep your text response concise and reference the panels.
    - ALWAYS return valid JSON. No text outside the JSON object.
""")


class OpenAIBackend:
    """Uses the OpenAI chat completions API with JSON mode."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        try:
            import openai
        except ImportError:
            raise ImportError("pip install openai")
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def respond(
        self,
        message: str,
        history: List[Dict[str, str]],
        contexts: List[Tuple[str, float]],
    ) -> AIResponse:
        messages = [{"role": "system", "content": _OPENAI_SYSTEM}]
        for m in history[-20:]:
            role = m.get("role", "user")
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": m.get("content", "")})
        messages.append({"role": "user", "content": message})

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.7,
                max_tokens=2048,
            )
            raw = resp.choices[0].message.content or "{}"
            data = json.loads(raw)
        except Exception as e:
            return AIResponse(
                text=f"OpenAI error: {e}. Falling back to built-in mode.",
                panels={},
                detected_contexts=[c for c, _ in contexts],
            )

        panels: Dict[str, PanelPayload] = {}
        for key in ("code", "data", "terminal", "preview", "chart"):
            p = data.get("panels", {}).get(key, {})
            if isinstance(p, dict) and p.get("visible"):
                content = p.get("content", "")
                lang = p.get("language", "python") if key == "code" else ""
                if key == "chart":
                    fig = _try_generate_chart(content if isinstance(content, str) else "bar")
                    if fig:
                        panels[key] = PanelPayload(visible=True, content=fig)
                elif key == "data" and isinstance(content, dict):
                    panels[key] = PanelPayload(visible=True, content=content)
                else:
                    panels[key] = PanelPayload(
                        visible=True,
                        content=content,
                        language=lang,
                    )

        return AIResponse(
            text=data.get("response", ""),
            panels=panels,
            detected_contexts=[c for c, _ in contexts],
        )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def get_ai_response(
    message: str,
    history: List[Dict[str, str]],
    api_key: str = "",
) -> AIResponse:
    """Unified entry point: pick backend, detect context, return response."""
    contexts = detect_contexts(message, history)

    if api_key.strip().startswith("sk-"):
        try:
            backend = OpenAIBackend(api_key=api_key.strip())
            return backend.respond(message, history, contexts)
        except Exception:
            pass

    return BuiltInBackend().respond(message, history, contexts)
