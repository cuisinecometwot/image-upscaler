import { useCallback, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { open } from "@tauri-apps/plugin-dialog";

type Status = "queued" | "running" | "done" | "error" | "cancelled";

interface Item {
  path: string;
  name: string;
  size: number;
  status: Status;
  progress: number;
  message?: string;
}

interface FileInfo {
  path: string;
  name: string;
  size: number;
}

interface FileStatus {
  index: number;
  status: Exclude<Status, "queued">;
  progress: number;
  message: string | null;
  output: string | null;
}

interface JobFinished {
  done: number;
  failed: number;
  cancelled: boolean;
}

const IMAGE_EXTS = ["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"];

function formatSize(bytes: number): string {
  if (bytes >= 1 << 20) return `${(bytes / (1 << 20)).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

function statusText(item: Item): string {
  switch (item.status) {
    case "queued":
      return "Waiting";
    case "running":
      return `${Math.round(item.progress)}%`;
    case "done":
      return "Done";
    case "error":
      return "Error";
    case "cancelled":
      return "Cancelled";
  }
}

export default function App() {
  const [items, setItems] = useState<Item[]>([]);
  const [outDir, setOutDir] = useState("");
  const [scale, setScale] = useState<"x2" | "x4">("x2");
  const [mode, setMode] = useState<"normal" | "anime">("normal");
  const [running, setRunning] = useState(false);
  const [summary, setSummary] = useState("");
  const [dragging, setDragging] = useState(false);

  const addPaths = useCallback(async (paths: string[]) => {
    if (paths.length === 0) return;
    const found = await invoke<FileInfo[]>("expand_inputs", { paths });
    if (found.length === 0) {
      setSummary("No supported images found (png, jpg, webp, bmp, tif).");
      return;
    }
    setSummary("");
    setItems((prev) => {
      const known = new Set(prev.map((i) => i.path));
      const fresh = found
        .filter((f) => !known.has(f.path))
        .map((f) => ({ ...f, status: "queued" as Status, progress: 0 }));
      return [...prev, ...fresh];
    });
    const suggested = await invoke<string>("default_output_dir", { firstInput: found[0].path });
    setOutDir((prev) => prev || suggested);
  }, []);

  useEffect(() => {
    const unlisteners = [
      listen<FileStatus>("file-status", ({ payload }) => {
        setItems((prev) =>
          prev.map((item, i) =>
            i === payload.index
              ? {
                  ...item,
                  status: payload.status,
                  progress: payload.progress,
                  message: payload.message ?? undefined,
                }
              : item,
          ),
        );
      }),
      listen<JobFinished>("job-finished", ({ payload }) => {
        setRunning(false);
        const parts = [`${payload.done} done`];
        if (payload.failed) parts.push(`${payload.failed} failed`);
        if (payload.cancelled) parts.push("cancelled");
        setSummary(parts.join(", "));
      }),
      getCurrentWebview().onDragDropEvent((event) => {
        const { type } = event.payload;
        if (type === "enter" || type === "over") setDragging(true);
        else if (type === "leave") setDragging(false);
        else if (type === "drop") {
          setDragging(false);
          addPaths(event.payload.paths);
        }
      }),
    ];
    return () => {
      unlisteners.forEach((p) => p.then((off) => off()));
    };
  }, [addPaths]);

  async function pickFiles() {
    const picked = await open({
      multiple: true,
      filters: [{ name: "Images", extensions: IMAGE_EXTS }],
    });
    if (picked) addPaths(picked);
  }

  async function pickFolder() {
    const picked = await open({ directory: true });
    if (picked) addPaths([picked]);
  }

  async function pickOutDir() {
    const picked = await open({ directory: true, defaultPath: outDir || undefined });
    if (picked) setOutDir(picked);
  }

  async function start() {
    setSummary("");
    setItems((prev) =>
      prev.map((i) => ({ ...i, status: "queued", progress: 0, message: undefined })),
    );
    try {
      await invoke("start_job", { files: items.map((i) => i.path), outDir, mode, scale });
      setRunning(true);
    } catch (e) {
      setSummary(String(e));
    }
  }

  const finished = items.filter((i) => i.status !== "queued" && i.status !== "running").length;
  const current = items.find((i) => i.status === "running");
  const overall = items.length
    ? ((finished + (current?.progress ?? 0) / 100) / items.length) * 100
    : 0;

  return (
    <main className={dragging ? "dragging" : ""}>
      <section className="toolbar">
        <button onClick={pickFiles} disabled={running}>
          Add files…
        </button>
        <button onClick={pickFolder} disabled={running}>
          Add folder…
        </button>
        <button onClick={() => setItems([])} disabled={running || items.length === 0}>
          Clear
        </button>
        <span className="hint">or drop files / folders here</span>
      </section>

      <section className="list">
        {items.length === 0 ? (
          <div className="empty">No images yet</div>
        ) : (
          items.map((item, i) => (
            <div className={`row ${item.status}`} key={item.path} title={item.message ?? item.path}>
              <span className="name">{item.name}</span>
              <span className="size">{formatSize(item.size)}</span>
              <span className="status">
                {item.status === "running" && <progress max={100} value={item.progress} />}
                {statusText(item)}
              </span>
              <button
                className="remove"
                onClick={() => setItems((prev) => prev.filter((_, j) => j !== i))}
                disabled={running}
                aria-label={`Remove ${item.name}`}
              >
                ×
              </button>
              {item.status === "error" && item.message && (
                <div className="error-text">{item.message}</div>
              )}
            </div>
          ))
        )}
      </section>

      <section className="settings">
        <label className="out">
          <span>Save to</span>
          <input
            value={outDir}
            onChange={(e) => setOutDir(e.target.value)}
            placeholder="Choose an output folder"
            disabled={running}
          />
          <button onClick={pickOutDir} disabled={running}>
            Browse…
          </button>
        </label>

        <fieldset disabled={running}>
          <legend>Scale</legend>
          {(["x2", "x4"] as const).map((s) => (
            <label key={s}>
              <input type="radio" checked={scale === s} onChange={() => setScale(s)} /> {s}
            </label>
          ))}
        </fieldset>

        <fieldset disabled={running}>
          <legend>Mode</legend>
          <label>
            <input type="radio" checked={mode === "normal"} onChange={() => setMode("normal")} />{" "}
            Normal
          </label>
          <label>
            <input type="radio" checked={mode === "anime"} onChange={() => setMode("anime")} />{" "}
            Anime
          </label>
        </fieldset>
      </section>

      <section className="actions">
        {running ? (
          <button className="primary" onClick={() => invoke("cancel_job")}>
            Cancel
          </button>
        ) : (
          <button className="primary" onClick={start} disabled={!items.length || !outDir}>
            Start
          </button>
        )}
        <progress className="overall" max={100} value={running || finished ? overall : 0} />
        <span className="summary">{running ? `${finished} / ${items.length}` : summary}</span>
        <button
          onClick={() => invoke("open_folder", { path: outDir })}
          disabled={!outDir || running || !items.some((i) => i.status === "done")}
        >
          Open folder
        </button>
      </section>
    </main>
  );
}
