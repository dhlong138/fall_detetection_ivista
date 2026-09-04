"""Local browser UI for annotated videos and LLM-confirmed fall crops."""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


PAGE = """<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Xem lại phát hiện ngã</title><style>
*{box-sizing:border-box}body{margin:0;background:#101827;color:#e8edf7;font-family:Arial,sans-serif}
header{padding:18px 24px;border-bottom:1px solid #27344a;display:flex;justify-content:space-between;align-items:center}
h1{font-size:20px;margin:0}.sub{color:#9fb0c9;font-size:13px}.layout{display:grid;grid-template-columns:minmax(0,2.35fr) minmax(280px,1fr);gap:16px;padding:16px;min-height:calc(100vh - 68px)}
.panel{background:#182235;border:1px solid #2c3d59;border-radius:12px;padding:14px;min-width:0}h2{font-size:15px;margin:0 0 12px}.video-wrap{background:#070b12;border-radius:8px;overflow:hidden}video{width:100%;max-height:76vh;display:block;background:#000}select{width:100%;padding:9px;border-radius:7px;background:#101827;color:#e8edf7;border:1px solid #3b4d6d;margin-bottom:12px}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(118px,1fr));gap:10px;max-height:76vh;overflow:auto}.crop{padding:5px;background:#101827;border:1px solid #31425e;border-radius:8px;cursor:zoom-in}.crop img{width:100%;aspect-ratio:1;object-fit:cover;display:block;border-radius:4px}.crop span{display:block;font-size:11px;color:#b8c6d9;padding-top:5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.empty{color:#9fb0c9;font-size:14px}
#zoom{display:none;position:fixed;inset:0;background:#000c;z-index:3;align-items:center;justify-content:center;cursor:zoom-out;padding:24px}#zoom.open{display:flex}#zoom img{max-width:95vw;max-height:92vh;object-fit:contain;transform:scale(1);transition:transform .15s;cursor:zoom-in}#zoom img.big{transform:scale(1.55);cursor:zoom-out}@media(max-width:850px){.layout{grid-template-columns:1fr}.gallery{max-height:none}}
</style></head><body><header><div><h1>Xem lại phát hiện ngã</h1><div class="sub">Video có box màu • ảnh crop do LLM xác nhận • click ảnh để zoom</div></div><div class="sub" id="count"></div></header>
<main class="layout"><section class="panel"><h2>Video kết quả</h2><select id="runs" aria-label="Chọn lần chạy"></select><select id="videos" aria-label="Chọn video"></select><div class="video-wrap"><video id="player" controls preload="metadata"></video></div><p class="sub">Dùng thanh điều khiển của video để tua; cuộn/nhấn vào ảnh ngã để phóng to.</p></section><aside class="panel"><h2>Ảnh ngã đã được LLM xác nhận</h2><div id="gallery" class="gallery"></div></aside></main><div id="zoom"><img id="zoomImage" alt="Ảnh ngã phóng to"></div>
<script>
const runsSelect=document.querySelector('#runs'),videos=document.querySelector('#videos'),player=document.querySelector('#player'),gallery=document.querySelector('#gallery'),count=document.querySelector('#count');let selectedRun='';
async function load(){const data=await (await fetch('/api/results')).json();if(!data.runs.length){count.textContent='Chưa có lần chạy';return}if(!data.runs.some(r=>r.name===selectedRun))selectedRun=data.runs[0].name;runsSelect.innerHTML=data.runs.map(r=>`<option value="${r.name}">${r.name}</option>`).join('');runsSelect.value=selectedRun;const run=data.runs.find(r=>r.name===selectedRun);count.textContent=`${run.name} • ${run.videos.length} video • ${run.crops.length} ảnh ngã`;videos.innerHTML=run.videos.length?run.videos.map(v=>`<option value="/runs/${encodeURIComponent(run.name)}/${encodeURIComponent(v)}">${v}</option>`).join(''):'<option>Chưa có video kết quả</option>';if(run.videos.length&&player.dataset.run!==run.name){player.src=videos.value;player.dataset.run=run.name;player.load()}gallery.innerHTML=run.crops.length?run.crops.map(c=>`<button class="crop" data-src="/runs/${encodeURIComponent(run.name)}/${encodeURIComponent(c)}"><img loading="lazy" src="/runs/${encodeURIComponent(run.name)}/${encodeURIComponent(c)}" alt="${c}"><span>${c}</span></button>`).join(''):'<p class="empty">Chưa có ảnh ngã được LLM xác nhận.</p>'}
runsSelect.onchange=()=>{selectedRun=runsSelect.value;player.dataset.run='';load()};videos.onchange=()=>{player.src=videos.value;player.load()};const zoom=document.querySelector('#zoom'),zoomImage=document.querySelector('#zoomImage');let cropClickTimer;
gallery.onclick=e=>{const card=e.target.closest('.crop');if(!card)return;clearTimeout(cropClickTimer);cropClickTimer=setTimeout(()=>{zoomImage.src=card.dataset.src;zoomImage.className='';zoom.classList.add('open')},240)};
gallery.ondblclick=e=>{const card=e.target.closest('.crop');if(!card)return;clearTimeout(cropClickTimer);const match=card.dataset.src.match(/fall_frame_(\d+)_track_/);if(!match)return;const seconds=Number(match[1])/30;zoom.classList.remove('open');player.currentTime=seconds;player.play().catch(()=>{});player.scrollIntoView({behavior:'smooth',block:'center'})};
zoom.onclick=e=>{if(e.target===zoomImage){zoomImage.classList.toggle('big')}else{zoom.classList.remove('open')}};load();setInterval(load,5000);
</script></body></html>"""


PAGE = PAGE.replace(
    "</style>",
    """.fall-timeline{position:relative;height:26px;margin:10px 2px 0;border-radius:7px;background:#0b1321;border:1px solid #31425e;overflow:hidden;cursor:pointer}.fall-timeline-progress{position:absolute;inset:0 auto 0 0;width:0;background:#1e467055;pointer-events:none}.fall-marker{position:absolute;top:3px;bottom:3px;width:max(7px,.55%);transform:translateX(-50%);border:0;border-radius:3px;background:#ff5c75;box-shadow:0 0 0 1px #ffd2da;cursor:pointer;padding:0}.fall-marker:hover,.fall-marker:focus{background:#ffd166;outline:2px solid #fff;z-index:1}.timeline-label{margin:6px 0 0;font-size:12px;color:#9fb0c9}</style>""",
)
PAGE = PAGE.replace(
    '<div class="video-wrap"><video id="player" controls preload="metadata"></video></div>',
    '<div class="video-wrap"><video id="player" controls preload="metadata"></video></div><div id="fallTimeline" class="fall-timeline" title="Click a checkpoint to seek"><div id="timelineProgress" class="fall-timeline-progress"></div></div><p id="timelineLabel" class="timeline-label">Red checkpoints mark LLM-confirmed falls.</p>',
)
PAGE = PAGE.replace(
    "</body>",
    """<script>
(() => {
  const timeline = document.querySelector('#fallTimeline');
  const progress = document.querySelector('#timelineProgress');
  const label = document.querySelector('#timelineLabel');
  const parseFrame = src => { const match = src.match(/fall_frame_(\\d+)_track_/); return match ? Number(match[1]) : null; };
  const seek = seconds => { player.currentTime = seconds; player.play().catch(() => {}); };
  const drawCheckpoints = () => {
    const duration = player.duration;
    if (!Number.isFinite(duration) || duration <= 0) { label.textContent = 'Đang tải các mốc cảnh ngã…'; return; }
    const checkpoints = [...gallery.querySelectorAll('.crop')].map(card => {
      const frame = parseFrame(card.dataset.src); return frame === null ? null : { frame, seconds: frame / 30 };
    }).filter(item => item && item.seconds <= duration);
    timeline.replaceChildren(progress, ...checkpoints.map(item => {
      const marker = document.createElement('button');
      marker.className = 'fall-marker';
      marker.style.left = `${Math.min(100, item.seconds / duration * 100)}%`;
      marker.title = `Mốc ngã: frame ${item.frame} (${item.seconds.toFixed(1)} giây)`;
      marker.setAttribute('aria-label', marker.title);
      marker.onclick = event => { event.stopPropagation(); seek(item.seconds); };
      return marker;
    }));
    label.textContent = checkpoints.length ? `${checkpoints.length} checkpoint ngã đã được LLM xác nhận — click mốc đỏ để tua.` : 'Chưa có checkpoint ngã được LLM xác nhận.';
  };
  player.addEventListener('loadedmetadata', drawCheckpoints);
  player.addEventListener('timeupdate', () => { progress.style.width = player.duration ? `${player.currentTime / player.duration * 100}%` : '0%'; });
  timeline.onclick = event => { if (event.target !== timeline && event.target !== progress) return; const rect = timeline.getBoundingClientRect(); seek((event.clientX - rect.left) / rect.width * player.duration); };
  const previousLoad = load;
  load = async () => { await previousLoad(); drawCheckpoints(); };
  load();
})();
</script></body>""",
)


def list_files(directory: Path, suffixes: set[str]) -> list[str]:
    if not directory.exists():
        return []
    return [
        path.name
        for path in sorted(directory.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if path.is_file() and path.suffix.lower() in suffixes and not path.name.startswith("h264_probe")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the local fall-review UI.")
    parser.add_argument("--runs-dir", default="output_runs")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    runs_dir = Path(args.runs_dir).resolve()

    def runs() -> list[dict[str, object]]:
        if not runs_dir.exists():
            return []
        directories = [path for path in runs_dir.iterdir() if path.is_dir()]
        return [
            {"name": path.name, "videos": list_files(path, {".mp4", ".avi", ".mov"}), "crops": list_files(path, {".jpg", ".jpeg", ".png"})}
            for path in sorted(directories, key=lambda item: item.stat().st_mtime, reverse=True)
        ]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args) -> None:
            pass

        def send_file(self, directory: Path, name: str) -> None:
            path = (directory / name).resolve()
            if directory not in path.parents or not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            size = path.stat().st_size
            start, end = 0, size - 1
            status = HTTPStatus.OK
            range_header = self.headers.get("Range", "")
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if match:
                start_text, end_text = match.groups()
                if start_text:
                    start = int(start_text)
                    end = int(end_text) if end_text else end
                elif end_text:
                    start = max(0, size - int(end_text))
                if start >= size or start > end:
                    self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    return
                end = min(end, size - 1)
                status = HTTPStatus.PARTIAL_CONTENT
            length = end - start + 1
            self.send_response(status)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as source:
                source.seek(start)
                remaining = length
                while remaining:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            path = urlparse(self.path).path
            if path == "/":
                body = PAGE.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers(); self.wfile.write(body); return
            if path == "/api/results":
                body = json.dumps({"runs": runs()}).encode()
                self.send_response(HTTPStatus.OK); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
            if path.startswith("/runs/"):
                parts = [unquote(part) for part in path.removeprefix("/runs/").split("/")]
                if len(parts) == 2:
                    run_directory = (runs_dir / parts[0]).resolve()
                    if runs_dir in run_directory.parents:
                        self.send_file(run_directory, parts[1]); return
            self.send_error(HTTPStatus.NOT_FOUND)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Open http://127.0.0.1:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
