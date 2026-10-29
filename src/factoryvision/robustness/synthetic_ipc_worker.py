"""Fixed child entry point for owned synthetic pixels, never actual model execution."""

import os
import sys
import time
from contextlib import suppress

from PIL import Image, PngImagePlugin

from factoryvision.robustness import synthetic_ipc as wire

FAULTS = (
    "none",
    "hang",
    "partial",
    "malformed",
    "oversized",
    "exit",
    "stale",
    "duplicate",
    "trailing",
)


def _deny_io(event, args):
    if event in (
        "open",
        "os.listdir",
        "os.scandir",
        "os.system",
        "subprocess.Popen",
    ) or event.startswith("socket."):
        raise RuntimeError("Synthetic worker I/O denied")
    if event == "import" and (
        args[0].startswith("torch") or args[0].startswith("factoryvision.inference.service")
    ):
        raise RuntimeError("Neural execution denied")


def main():
    wire.check(len(sys.argv) == 2 and sys.argv[1] in FAULTS)
    wire.check(PngImagePlugin is not None)
    Image.preinit()  # Pillow's import-only codec registration; no image evaluation/warmup.
    wire.check(
        not any(
            name == "torch" or name == "factoryvision.inference.service" for name in sys.modules
        )
    )
    session = wire.WorkerSession()
    sys.addaudithook(_deny_io)
    reader, writer = sys.stdin.buffer, sys.stdout.buffer

    def send(message):
        writer.write(wire.encode(message))
        writer.flush()

    seq = 0
    try:
        send(wire.envelope("ready", 0, status="ready"))
        for seq in range(1, 6):
            data = wire.decode(wire.read_frame(reader.read, wire.MAX_JSON))
            wire.validate(data, "inspect", seq)
            payload = wire.read_frame(reader.read, wire.MAX_PNG)
            # Finite, reviewed synthetic fault probes; never selected by IPC input.
            if seq == 2:
                fault = sys.argv[1]
                if fault == "hang":
                    while True:
                        time.sleep(0.05)
                if fault == "exit":
                    os._exit(97)
                if fault in ("partial", "oversized", "malformed"):
                    writer.write(
                        {
                            "partial": b"\x00\x00\x00\x20{",
                            "oversized": b"\xff\xff\xff\xff",
                            "malformed": wire.frame(b'{"x":NaN}', wire.MAX_JSON),
                        }[fault]
                    )
                    writer.flush()
                    return 31
            result = session.inspect(data, payload)
            if seq == 2 and sys.argv[1] == "stale":
                result["seq"] = 1
            send(result)
            if seq == 2 and sys.argv[1] == "duplicate":
                send(result)
        data = wire.decode(wire.read_frame(reader.read, wire.MAX_JSON))
        wire.validate(data, "close", 6)
        send(wire.envelope("done", 6, status="complete"))
        if sys.argv[1] == "trailing":
            send(wire.envelope("done", 6, status="complete"))
        return 0
    except BaseException:
        with suppress(BaseException):
            send(wire.envelope("error", seq, status="failed", code="invalid_request"))
        return 31


if __name__ == "__main__":
    try:
        code = main()
    except BaseException:
        code = 31
    sys.exit(code)
