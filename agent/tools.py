"""Tool registry shared by the agent orchestrator (agent/orchestrator.py) and
the synthetic tool-use training examples (data/build_tool_examples.py, wired
into data/prepare_instruct.py) -- kept in one place so the tools the model is
trained to call and the tools the orchestrator can actually execute never
drift apart.

Tool-call convention: plain text, not new tokenizer vocab. The model emits

    <tool_call>{"name": "...", "args": {...}}</tool_call>

inside its assistant turn; the orchestrator parses that, runs the matching
function below, and feeds the result back as

    <tool_result>...</tool_result>

so the model can keep generating a natural-language reply. These are ordinary
subword text, not special tokens, so no tokenizer retrain is needed to teach
this -- it's purely an instruction-finetuning-data problem.
"""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tokenizer.special_tokens import DEFAULT_SYSTEM_PROMPT  # noqa: E402

STATE_DIR = Path(__file__).resolve().parent / "state"
REMINDERS_PATH = STATE_DIR / "reminders.json"

# Deliberately an allowlist, not arbitrary shell exec: Shannon is a shared,
# no-sudo machine, and a headless training container can't reach a real
# GUI/audio device anyway, so "device control" here means read-only container
# introspection, not actually controlling anything. Real device control
# (open app X, play media) belongs in a client-side orchestrator running ON
# the target device once the model is exported for that -- see
# [[project-malayalam-llm-vision]] for the cross-device plan.
SYSTEM_INFO_COMMANDS = {
    "disk_usage": ["df", "-h", "/"],
    "uptime": ["uptime"],
    "memory": ["free", "-h"],
}


def _load_reminders():
    if not REMINDERS_PATH.exists():
        return []
    return json.loads(REMINDERS_PATH.read_text(encoding="utf-8"))


def _save_reminders(items):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    REMINDERS_PATH.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")


def tool_get_datetime(args):
    now = datetime.now().astimezone()
    return json.dumps({"datetime": now.strftime("%Y-%m-%d %H:%M:%S %Z"), "weekday": now.strftime("%A")})


def tool_calculator(args):
    expr = str(args.get("expression", ""))
    allowed = set("0123456789+-*/(). %")
    if not expr or not set(expr) <= allowed:
        return json.dumps({"error": "expression must be plain arithmetic (digits, + - * / ( ) . %)"})
    try:
        # eval is safe here only because of the strict character allowlist
        # above -- no names, no attribute access, no builtins reachable.
        result = eval(expr, {"__builtins__": {}}, {})
    except Exception as e:
        return json.dumps({"error": str(e)})
    return json.dumps({"result": result})


_UNIT_FACTORS = {
    ("km", "mi"): 0.621371, ("mi", "km"): 1.60934,
    ("kg", "lb"): 2.20462, ("lb", "kg"): 0.453592,
}


def tool_unit_convert(args):
    from_unit = str(args.get("from", "")).lower()
    to_unit = str(args.get("to", "")).lower()
    try:
        value = float(args.get("value"))
    except (TypeError, ValueError):
        return json.dumps({"error": "value must be numeric"})

    if from_unit == "c" and to_unit == "f":
        return json.dumps({"result": value * 9 / 5 + 32})
    if from_unit == "f" and to_unit == "c":
        return json.dumps({"result": (value - 32) * 5 / 9})

    factor = _UNIT_FACTORS.get((from_unit, to_unit))
    if factor is None:
        return json.dumps({"error": f"unsupported conversion {from_unit}->{to_unit}"})
    return json.dumps({"result": value * factor})


def tool_set_reminder(args):
    text = str(args.get("text", "")).strip()
    when = str(args.get("when", "")).strip()
    if not text:
        return json.dumps({"error": "text is required"})
    items = _load_reminders()
    items.append({"text": text, "when": when, "created": datetime.now(timezone.utc).isoformat()})
    _save_reminders(items)
    return json.dumps({"status": "saved", "count": len(items)})


def tool_list_reminders(args):
    return json.dumps({"reminders": _load_reminders()})


def tool_system_info(args):
    which = str(args.get("which", "")).lower()
    cmd = SYSTEM_INFO_COMMANDS.get(which)
    if cmd is None:
        return json.dumps({"error": f"unknown info type, choose one of {list(SYSTEM_INFO_COMMANDS)}"})
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
    except Exception as e:
        return json.dumps({"error": str(e)})
    return json.dumps({"output": out.strip()})


TOOLS = {
    "get_datetime": {
        "fn": tool_get_datetime,
        "description": "ഇപ്പോഴത്തെ തീയതിയും സമയവും അറിയാൻ.",
        "args": {},
    },
    "calculator": {
        "fn": tool_calculator,
        "description": "ഗണിത കണക്കുകൂട്ടലുകൾക്ക് (കൂട്ടൽ, കുറയ്ക്കൽ, ഗുണനം, ഹരണം).",
        "args": {"expression": "string, e.g. '15 + 27'"},
    },
    "unit_convert": {
        "fn": tool_unit_convert,
        "description": "അളവ് യൂണിറ്റുകൾ മാറ്റാൻ (km/mi, kg/lb, c/f).",
        "args": {"value": "number", "from": "string", "to": "string"},
    },
    "set_reminder": {
        "fn": tool_set_reminder,
        "description": "ഒരു ഓർമ്മപ്പെടുത്തൽ സൂക്ഷിക്കാൻ.",
        "args": {"text": "string", "when": "string, optional"},
    },
    "list_reminders": {
        "fn": tool_list_reminders,
        "description": "സൂക്ഷിച്ച എല്ലാ ഓർമ്മപ്പെടുത്തലുകളും കാണിക്കാൻ.",
        "args": {},
    },
    "system_info": {
        "fn": tool_system_info,
        "description": "സെർവറിന്റെ അവസ്ഥ അറിയാൻ (disk_usage, uptime, memory).",
        "args": {"which": "string, one of disk_usage/uptime/memory"},
    },
}


def execute_tool(name, args):
    tool = TOOLS.get(name)
    if tool is None:
        return json.dumps({"error": f"unknown tool '{name}', available: {list(TOOLS)}"})
    try:
        return tool["fn"](args or {})
    except Exception as e:
        return json.dumps({"error": f"tool '{name}' failed: {e}"})


def tool_list_prompt_block():
    """Malayalam tool listing injected into the system prompt -- generated
    from TOOLS so the description the model is trained/prompted with can
    never describe a different tool set than execute_tool() actually runs."""
    lines = ["നിനക്ക് ഈ ടൂളുകൾ ഉപയോഗിക്കാം. ടൂൾ വേണമെങ്കിൽ കൃത്യമായി ഇങ്ങനെ എഴുതുക:",
             '<tool_call>{"name": "ടൂൾ_പേര്", "args": {...}}</tool_call>',
             "ലഭ്യമായ ടൂളുകൾ:"]
    for name, spec in TOOLS.items():
        args_desc = ", ".join(f"{k} ({v})" for k, v in spec["args"].items()) or "ആർഗ്യുമെന്റ് ഇല്ല"
        lines.append(f"- {name}: {spec['description']} ആർഗ്യുമെന്റുകൾ: {args_desc}")
    return "\n".join(lines)


AGENT_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT + "\n\n" + tool_list_prompt_block()


# ---- synthetic training examples --------------------------------------
# A handful of phrasings per tool so the model sees the <tool_call> pattern
# in varied contexts rather than memorizing one instruction per tool.
_EXAMPLES = [
    ("ഇന്നത്തെ തീയതി എന്താണ്?", "get_datetime", {}),
    ("ഇപ്പോൾ എത്ര മണിയായി?", "get_datetime", {}),
    ("15 ഉം 27 ഉം കൂട്ടിയാൽ എത്ര വരും?", "calculator", {"expression": "15 + 27"}),
    ("120 നെ 8 കൊണ്ട് ഹരിച്ചാൽ എത്ര?", "calculator", {"expression": "120 / 8"}),
    ("9 ന്റെ 7 ഇരട്ടി എത്രയാണ്?", "calculator", {"expression": "9 * 7"}),
    ("10 കിലോമീറ്റർ എത്ര മൈൽ ആണ്?", "unit_convert", {"value": 10, "from": "km", "to": "mi"}),
    ("30 ഡിഗ്രി സെൽഷ്യസ് എത്ര ഫാരൻഹീറ്റ് ആണ്?", "unit_convert", {"value": 30, "from": "c", "to": "f"}),
    ("5 കിലോഗ്രാം എത്ര പൗണ്ട് ആണ്?", "unit_convert", {"value": 5, "from": "kg", "to": "lb"}),
    ("നാളെ രാവിലെ 9 മണിക്ക് ഡോക്ടറെ കാണാൻ ഓർമ്മിപ്പിക്കുക", "set_reminder",
     {"text": "ഡോക്ടറെ കാണാൻ", "when": "നാളെ രാവിലെ 9 മണി"}),
    ("വൈകുന്നേരം അമ്മയെ വിളിക്കാൻ ഓർമ്മപ്പെടുത്തുക", "set_reminder",
     {"text": "അമ്മയെ വിളിക്കാൻ", "when": "വൈകുന്നേരം"}),
    ("എന്റെ ഓർമ്മപ്പെടുത്തലുകൾ എന്തൊക്കെയാണ്?", "list_reminders", {}),
    ("സെർവറിൽ എത്ര ഡിസ്ക് സ്ഥലം ബാക്കിയുണ്ട്?", "system_info", {"which": "disk_usage"}),
    ("സിസ്റ്റത്തിന്റെ മെമ്മറി ഉപയോഗം എത്രയാണ്?", "system_info", {"which": "memory"}),
]


def build_tool_training_examples():
    """Returns {instruction, input, output, source} records in the same
    shape data/prepare_instruct.py's other loaders use, demonstrating the
    <tool_call> convention for every tool in TOOLS."""
    records = []
    for instruction, tool_name, tool_args in _EXAMPLES:
        assert tool_name in TOOLS, f"{tool_name} not in TOOLS -- keep _EXAMPLES in sync"
        call = json.dumps({"name": tool_name, "args": tool_args}, ensure_ascii=False)
        records.append({
            "instruction": instruction,
            "input": "",
            "output": f"<tool_call>{call}</tool_call>",
            "source": "tool_examples",
        })
    return records
