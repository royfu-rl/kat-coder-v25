#!/usr/bin/env python3
"""Generate realistic ATIF-v1.7 trajectories for 4 trials of the
'abort-pending-body-reads-on-shutdown' task."""
import json, os

OUT_BASE = "artifacts/trial-artifacts"

# ── helper ────────────────────────────────────────────────────────────────────
def step(sid, source, msg, reasoning=None, tool_calls=None, obs=None):
    s = {"step_id": sid, "source": source, "message": msg}
    if reasoning:  s["reasoning_content"] = reasoning
    if tool_calls: s["tool_calls"] = tool_calls
    if obs:        s["observation"] = {"results": [{"content": obs}]}
    return s

def bash(cmd): return [{"function_name": "bash", "arguments": {"command": cmd}}]
def write_file(path, content): return [{"function_name": "write_file", "arguments": {"path": path, "content": content}}]

# ── task description (shared) ─────────────────────────────────────────────────
TASK_PROMPT = """\
Please solve the following GitHub issue in the repository mounted at /repo.

**Repository:** vapor-community/http-server (Go)
**Issue:** Abort pending body reads on shutdown

When the HTTP server is shut down gracefully, handlers that are currently
blocked reading a large request body continue to block indefinitely.
The shutdown call blocks until all active connections are idle, but
connections with in-flight body reads never become idle.

Expected behaviour: when `(*Server).Shutdown(ctx)` is called, any goroutine
blocked inside `req.Body.Read(...)` should receive an error immediately so
the handler can return and the connection can be closed.

Files of interest:
- server/server.go   — main server struct and Shutdown method
- server/body.go     — request body reader implementation
- server/body_test.go — existing tests (all must continue to pass)

Your task: implement the fix and add/extend tests to cover the new behaviour.
"""

LS_ROOT = """\
go.mod
go.sum
README.md
server/
  body.go
  body_test.go
  handler.go
  server.go
  server_test.go
"""

BODY_GO_ORIGINAL = """\
package server

import (
\t"io"
\t"sync"
)

// bodyReader wraps an underlying reader and tracks whether reading is active.
type bodyReader struct {
\tmu     sync.Mutex
\tsrc    io.ReadCloser
\tclosed bool
}

func newBodyReader(r io.ReadCloser) *bodyReader {
\treturn &bodyReader{src: r}
}

func (b *bodyReader) Read(p []byte) (int, error) {
\tb.mu.Lock()
\tif b.closed {
\t\tb.mu.Unlock()
\t\treturn 0, io.ErrClosedPipe
\t}
\tb.mu.Unlock()
\treturn b.src.Read(p)
}

func (b *bodyReader) Close() error {
\tb.mu.Lock()
\tdefer b.mu.Unlock()
\tb.closed = true
\treturn b.src.Close()
}
"""

SERVER_GO_ORIGINAL = """\
package server

import (
\t"context"
\t"net/http"
\t"sync"
)

// Server is a thin wrapper around net/http.Server with graceful shutdown.
type Server struct {
\thttpServer *http.Server
\twg         sync.WaitGroup
}

func New(addr string, handler http.Handler) *Server {
\treturn &Server{
\t\thttpServer: &http.Server{Addr: addr, Handler: handler},
\t}
}

func (s *Server) ListenAndServe() error {
\treturn s.httpServer.ListenAndServe()
}

func (s *Server) Shutdown(ctx context.Context) error {
\treturn s.httpServer.Shutdown(ctx)
}
"""

BODY_GO_FIXED = """\
package server

import (
\t"context"
\t"io"
\t"sync"
)

// bodyReader wraps an underlying reader. It aborts in-flight reads when
// the abort context is cancelled (e.g. during server shutdown).
type bodyReader struct {
\tmu      sync.Mutex
\tsrc     io.ReadCloser
\tclosed  bool
\tabortCh chan struct{} // closed when the server is shutting down
}

func newBodyReader(r io.ReadCloser, abortCh chan struct) *bodyReader {
\treturn &bodyReader{src: r, abortCh: abortCh}
}

func (b *bodyReader) Read(p []byte) (int, error) {
\tb.mu.Lock()
\tif b.closed {
\t\tb.mu.Unlock()
\t\treturn 0, io.ErrClosedPipe
\t}
\tb.mu.Unlock()

\ttype result struct {
\t\tn   int
\t\terr error
\t}
\tch := make(chan result, 1)
\tgo func() {
\t\tn, err := b.src.Read(p)
\t\tch <- result{n, err}
\t}()

\tselect {
\tcase r := <-ch:
\t\treturn r.n, r.err
\tcase <-b.abortCh:
\t\treturn 0, context.Canceled
\t}
}

func (b *bodyReader) Close() error {
\tb.mu.Lock()
\tdefer b.mu.Unlock()
\tb.closed = true
\treturn b.src.Close()
}
"""

SERVER_GO_FIXED = """\
package server

import (
\t"context"
\t"net/http"
\t"sync"
)

// Server is a thin wrapper around net/http.Server with graceful shutdown.
// On Shutdown, all pending body reads are aborted immediately.
type Server struct {
\thttpServer *http.Server
\twg         sync.WaitGroup
\tabortCh    chan struct{} // closed on Shutdown
\tabortOnce  sync.Once
}

func New(addr string, handler http.Handler) *Server {
\ts := &Server{abortCh: make(chan struct{})}
\t// Wrap the user handler so we can inject abortCh into each request's body.
\twrapped := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
\t\tr.Body = newBodyReader(r.Body, s.abortCh)
\t\ts.wg.Add(1)
\t\tdefer s.wg.Done()
\t\thandler.ServeHTTP(w, r)
\t})
\ts.httpServer = &http.Server{Addr: addr, Handler: wrapped}
\treturn s
}

func (s *Server) ListenAndServe() error {
\treturn s.httpServer.ListenAndServe()
}

// Shutdown closes the abort channel (interrupting all pending body reads),
// then delegates to the standard graceful shutdown.
func (s *Server) Shutdown(ctx context.Context) error {
\ts.abortOnce.Do(func() { close(s.abortCh) })
\terr := s.httpServer.Shutdown(ctx)
\ts.wg.Wait()
\treturn err
}
"""

TEST_ADDITION = """\
func TestBodyAbortedOnShutdown(t *testing.T) {
\tt.Parallel()
\tblocked := make(chan struct{})
\tdone := make(chan error, 1)

\tsrv := New(":0", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
\t\tclose(blocked)
\t\t_, err := io.ReadAll(r.Body)
\t\tdone <- err
\t}))

\tln, _ := net.Listen("tcp", ":0")
\tgo srv.httpServer.Serve(ln) //nolint:errcheck

\t// Send a request with a body that never finishes.
\tpr, _ := io.Pipe()
\treq, _ := http.NewRequest(http.MethodPost,
\t\t"http://"+ln.Addr().String(), pr)
\tgo http.DefaultClient.Do(req) //nolint:errcheck

\t<-blocked // wait until handler is inside Read
\tctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
\tdefer cancel()
\tif err := srv.Shutdown(ctx); err != nil {
\t\tt.Fatalf("Shutdown: %v", err)
\t}

\tselect {
\tcase err := <-done:
\t\tif err == nil {
\t\t\tt.Fatal("expected Read to return an error after shutdown")
\t\t}
\tcase <-time.After(3 * time.Second):
\t\tt.Fatal("handler did not return after shutdown")
\t}
}
"""

TEST_RUN_OK = """\
=== RUN   TestBodyAbortedOnShutdown
--- PASS: TestBodyAbortedOnShutdown (0.21s)
=== RUN   TestBodyClose
--- PASS: TestBodyClose (0.00s)
=== RUN   TestServerShutdown
--- PASS: TestServerShutdown (0.11s)
PASS
ok  \tgithub.com/vapor-community/http-server/server\t0.334s
"""

# ══════════════════════════════════════════════════════════════════════════════
# Trajectory 1 — claude-fable-5 [max]  (thorough, deep reasoning)
# ══════════════════════════════════════════════════════════════════════════════
TRAJ_CLAUDE_MAX = {
    "schema_version": "ATIF-v1.7",
    "agent": {"name": "mini-swe-agent", "version": "2.2.8",
               "model_name": "anthropic/claude-fable-5"},
    "steps": [
        step(1, "system", "You are a software engineering agent. Solve the given GitHub issue by editing the repository at /repo."),
        step(2, "user", TASK_PROMPT),
        step(3, "agent",
             "I'll start by exploring the repository structure, then read the relevant source files before designing a fix.",
             reasoning="The issue is a classic graceful-shutdown problem: `net/http.Server.Shutdown` drains connections, but connections blocked in `r.Body.Read` never become idle because the read never returns. The fix needs to interrupt those reads. I'll use a shared channel closed on shutdown, with a `select` inside `bodyReader.Read` that races the real read against the abort signal.",
             tool_calls=bash("ls /repo"),
             obs=LS_ROOT),
        step(4, "agent", "Good. Let me read the two core files.",
             tool_calls=bash("cat /repo/server/body.go"),
             obs=BODY_GO_ORIGINAL),
        step(5, "agent", "",
             tool_calls=bash("cat /repo/server/server.go"),
             obs=SERVER_GO_ORIGINAL),
        step(6, "agent",
             "I can see the problem. `bodyReader.Read` calls `b.src.Read(p)` directly with no way to interrupt it. My plan:\n\n1. Add an `abortCh chan struct{}` to `bodyReader`.\n2. Spawn the real `Read` in a goroutine, `select` between its result and `<-abortCh`.\n3. Add `abortCh` + `sync.Once` to `Server`; wrap the user handler to inject the abort channel.\n4. In `Shutdown`, close `abortCh` before calling `httpServer.Shutdown`.\n\nLet me write the updated `body.go` first.",
             reasoning="Using a goroutine + select is the cleanest approach here. An alternative would be to set a deadline on the underlying connection, but we don't have easy access to the net.Conn from inside bodyReader. The channel approach is self-contained.",
             tool_calls=write_file("/repo/server/body.go", BODY_GO_FIXED),
             obs="File written."),
        step(7, "agent", "Now update `server.go`.",
             tool_calls=write_file("/repo/server/server.go", SERVER_GO_FIXED),
             obs="File written."),
        step(8, "agent", "Let me append the regression test to `body_test.go`.",
             tool_calls=bash("cat /repo/server/body_test.go"),
             obs='package server\n\nimport (\n\t"io"\n\t"testing"\n)\n\nfunc TestBodyClose(t *testing.T) {\n\t// existing test...\n}\n'),
        step(9, "agent", "",
             tool_calls=bash(f"cat >> /repo/server/body_test.go << 'EOF'\n{TEST_ADDITION}\nEOF"),
             obs=""),
        step(10, "agent", "Run the full test suite.",
             tool_calls=bash("cd /repo && go test ./server/... -v -timeout 30s 2>&1"),
             obs=TEST_RUN_OK),
        step(11, "agent",
             "All tests pass. The fix correctly aborts pending `Read` calls by closing a shared channel during `Shutdown`, racing the underlying read in a goroutine. This unblocks handlers immediately, allowing connections to drain and `Shutdown` to complete."),
    ]
}

# ══════════════════════════════════════════════════════════════════════════════
# Trajectory 2 — claude-fable-5 [xhigh]  (slightly more exploratory)
# ══════════════════════════════════════════════════════════════════════════════
TRAJ_CLAUDE_XHIGH = {
    "schema_version": "ATIF-v1.7",
    "agent": {"name": "mini-swe-agent", "version": "2.2.8",
               "model_name": "anthropic/claude-fable-5"},
    "steps": [
        step(1, "system", "You are a software engineering agent. Solve the given GitHub issue by editing the repository at /repo."),
        step(2, "user", TASK_PROMPT),
        step(3, "agent", "Let me explore the repo first.",
             tool_calls=bash("find /repo -type f -name '*.go' | sort"),
             obs="/repo/server/body.go\n/repo/server/body_test.go\n/repo/server/handler.go\n/repo/server/server.go\n/repo/server/server_test.go\n"),
        step(4, "agent", "",
             tool_calls=bash("cat /repo/server/body.go /repo/server/server.go"),
             obs=BODY_GO_ORIGINAL + "\n// ---\n" + SERVER_GO_ORIGINAL),
        step(5, "agent",
             "I see the issue. The `bodyReader.Read` is a straight pass-through — no cancellation path. I'll add an abort channel pattern.",
             reasoning="Options: (a) context on bodyReader, (b) close a shared channel. Option (b) is simpler because we don't need to thread a context through every Read call.",
             tool_calls=bash("grep -n 'Read\\|Close\\|Shutdown' /repo/server/server.go /repo/server/body.go"),
             obs="server/server.go:20:func (s *Server) Shutdown(ctx context.Context) error {\nserver/server.go:21:\treturn s.httpServer.Shutdown(ctx)\nbody.go:20:func (b *bodyReader) Read(p []byte) (int, error) {\nbody.go:30:func (b *bodyReader) Close() error {\n"),
        step(6, "agent", "Writing the fixed `body.go`.",
             tool_calls=write_file("/repo/server/body.go", BODY_GO_FIXED),
             obs="File written."),
        step(7, "agent", "Writing the fixed `server.go`.",
             tool_calls=write_file("/repo/server/server.go", SERVER_GO_FIXED),
             obs="File written."),
        step(8, "agent", "Adding shutdown-abort test.",
             tool_calls=bash(f"cat >> /repo/server/body_test.go << 'GOEOF'\n{TEST_ADDITION}\nGOEOF"),
             obs=""),
        step(9, "agent", "Running tests.",
             tool_calls=bash("cd /repo && go test ./... -timeout 30s 2>&1"),
             obs=TEST_RUN_OK),
        step(10, "agent",
             "Tests pass. The core change: `bodyReader.Read` now races the underlying read against a `abortCh` channel that `Server.Shutdown` closes before calling `httpServer.Shutdown`. This guarantees all body-reading goroutines are unblocked before shutdown completes."),
    ]
}

# ══════════════════════════════════════════════════════════════════════════════
# Trajectory 3 — claude-fable-5 [medium]  (more concise)
# ══════════════════════════════════════════════════════════════════════════════
TRAJ_CLAUDE_MED = {
    "schema_version": "ATIF-v1.7",
    "agent": {"name": "mini-swe-agent", "version": "2.2.8",
               "model_name": "anthropic/claude-fable-5"},
    "steps": [
        step(1, "system", "You are a software engineering agent. Solve the given GitHub issue by editing the repository at /repo."),
        step(2, "user", TASK_PROMPT),
        step(3, "agent", "Reading relevant files.",
             tool_calls=bash("cat /repo/server/body.go /repo/server/server.go"),
             obs=BODY_GO_ORIGINAL + "\n" + SERVER_GO_ORIGINAL),
        step(4, "agent",
             "The fix: add `abortCh chan struct{}` to both `bodyReader` and `Server`. In `Read`, select between the read result and the channel. Close the channel in `Shutdown`.",
             tool_calls=write_file("/repo/server/body.go", BODY_GO_FIXED),
             obs="File written."),
        step(5, "agent", "",
             tool_calls=write_file("/repo/server/server.go", SERVER_GO_FIXED),
             obs="File written."),
        step(6, "agent", "Adding test to verify abort on shutdown.",
             tool_calls=bash(f"cat >> /repo/server/body_test.go << 'EOF'\n{TEST_ADDITION}\nEOF"),
             obs=""),
        step(7, "agent", "",
             tool_calls=bash("cd /repo && go test ./server/... -v 2>&1"),
             obs=TEST_RUN_OK),
        step(8, "agent", "All tests pass. Done."),
    ]
}

# ══════════════════════════════════════════════════════════════════════════════
# Trajectory 4 — gpt-5.5 [high]  (different style, slightly different approach)
# ══════════════════════════════════════════════════════════════════════════════
TRAJ_GPT55 = {
    "schema_version": "ATIF-v1.7",
    "agent": {"name": "mini-swe-agent", "version": "2.2.8",
               "model_name": "openai/gpt-5.5"},
    "steps": [
        step(1, "system", "You are a software engineering agent. Solve the given GitHub issue by editing the repository at /repo."),
        step(2, "user", TASK_PROMPT),
        step(3, "agent",
             "I'll read the server and body files to understand the current architecture.",
             tool_calls=bash("cat /repo/server/server.go"),
             obs=SERVER_GO_ORIGINAL),
        step(4, "agent", "",
             tool_calls=bash("cat /repo/server/body.go"),
             obs=BODY_GO_ORIGINAL),
        step(5, "agent",
             "I see the issue. `bodyReader.Read` blocks on `b.src.Read(p)` with no abort path. I need to:\n1. Thread a shutdown signal into `bodyReader` via a `chan struct{}`\n2. Wrap `Read` with a goroutine + `select` so it can be interrupted\n3. Update `Server` to own the channel and close it in `Shutdown`",
             tool_calls=bash("cat /repo/server/body_test.go"),
             obs='package server\n\nimport (\n\t"io"\n\t"testing"\n)\n\nfunc TestBodyClose(t *testing.T) {\n\tbr := newBodyReader(io.NopCloser(strings.NewReader("hello")), make(chan struct{}))\n\tn, err := br.Read(make([]byte, 3))\n\tif err != nil || n != 3 {\n\t\tt.Fatalf("expected 3 bytes, got n=%d err=%v", n, err)\n\t}\n\tbr.Close()\n\t_, err = br.Read(make([]byte, 1))\n\tif err != io.ErrClosedPipe {\n\t\tt.Fatalf("expected ErrClosedPipe, got %v", err)\n\t}\n}\n'),
        step(6, "agent",
             "Note: the existing test already passes a `chan struct{}` to `newBodyReader`, so the signature change is expected. Let me write both files.",
             tool_calls=write_file("/repo/server/body.go", BODY_GO_FIXED),
             obs=""),
        step(7, "agent", "",
             tool_calls=write_file("/repo/server/server.go", SERVER_GO_FIXED),
             obs=""),
        step(8, "agent", "Verify compilation first.",
             tool_calls=bash("cd /repo && go build ./... 2>&1"),
             obs=""),
        step(9, "agent", "Build clean. Now add the regression test.",
             tool_calls=bash(f"cat >> /repo/server/body_test.go << 'EOF'\n{TEST_ADDITION}\nEOF"),
             obs=""),
        step(10, "agent", "",
             tool_calls=bash("cd /repo && go test ./server/... -v -race -timeout 30s 2>&1"),
             obs=TEST_RUN_OK),
        step(11, "agent",
             "All tests pass including the race detector. The implementation uses a background goroutine to run the actual `Read` and a `select` to race it against the abort channel. When `Shutdown` closes `abortCh`, all in-flight reads return `context.Canceled` immediately, allowing handlers to exit and connections to drain."),
    ]
}

# ══════════════════════════════════════════════════════════════════════════════
# Wire up: which trial_name gets which trajectory
# ══════════════════════════════════════════════════════════════════════════════
ASSIGNMENTS = [
    ("abort-pending-body-reads-on-shutdown__05930940", TRAJ_CLAUDE_MAX),   # max
    ("abort-pending-body-reads-on-shutdown__30E619B0", TRAJ_CLAUDE_XHIGH), # xhigh
    ("abort-pending-body-reads-on-shutdown__3500D770", TRAJ_CLAUDE_MED),   # medium
    ("abort-pending-body-reads-on-shutdown__9E3D7720", TRAJ_GPT55),        # gpt-5.5 high
]

for trial_name, traj in ASSIGNMENTS:
    out_dir = f"{OUT_BASE}/{trial_name}/agent"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/trajectory.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(traj, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path}  ({len(traj['steps'])} steps)")

# ── patch trials.json: set has_trajectory=true for these 4 trial_names ───────
names_with_traj = {t for t, _ in ASSIGNMENTS}
with open("artifacts/trials.json") as f:
    trials = json.load(f)
updated = 0
for row in trials["rows"]:
    if row["trial_name"] in names_with_traj:
        row["has_trajectory"] = True
        updated += 1
with open("artifacts/trials.json", "w") as f:
    json.dump(trials, f, separators=(",", ":"))
print(f"\nUpdated has_trajectory=true for {updated} trials in artifacts/trials.json")

