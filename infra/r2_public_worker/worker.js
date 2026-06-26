// Public read-only proxy for the `podcast` R2 bucket (dataset deliverable).
// Deployed via `wrangler deploy` -- see README in this directory for the exact command.
// Pattern follows Cloudflare's own R2-public-read Worker example (same shape as
// https://denoflare.dev/examples/r2-public-read), extended with an HTML directory
// listing (browse any "/"-suffixed path) since the dataset is meant to be explored,
// not just fetched by exact key. Object responses support range + conditional
// requests so audio clients can byte-range FLAC clips.

const PAGE_HEAD = `<!doctype html><html><head><meta charset="utf-8">
<title>Podcast Speech Dataset</title>
<style>
body{font-family:-apple-system,system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;line-height:1.5;color:#222}
a{color:#0645ad;text-decoration:none} a:hover{text-decoration:underline}
table{width:100%;border-collapse:collapse;margin-top:1rem}
td,th{text-align:left;padding:4px 8px;border-bottom:1px solid #eee}
td.size{color:#666;text-align:right;white-space:nowrap}
code{background:#f4f4f4;padding:2px 4px;border-radius:3px}
.crumbs{color:#666}
</style></head><body>`;
const PAGE_FOOT = `</body></html>`;

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function renderCrumbs(prefix) {
  const parts = prefix.split("/").filter(Boolean);
  const links = [`<a href="/">root</a>`];
  let path = "";
  for (const part of parts) {
    path += part + "/";
    links.push(`<a href="/${encodeURI(path)}">${escapeHtml(part)}</a>`);
  }
  return `<p class="crumbs">${links.join(" / ")}</p>`;
}

async function renderListing(env, prefix, request) {
  const url = new URL(request.url);
  const cursor = url.searchParams.get("cursor") || undefined;
  const listed = await env.BUCKET.list({ prefix, delimiter: "/", cursor, limit: 1000 });

  const rows = [];
  for (const p of listed.delimitedPrefixes) {
    const name = p.slice(prefix.length);
    rows.push(`<tr><td>\u{1F4C1} <a href="/${encodeURI(p)}">${escapeHtml(name)}</a></td><td class="size"></td></tr>`);
  }
  for (const obj of listed.objects) {
    const name = obj.key.slice(prefix.length);
    if (name === "") continue;
    rows.push(
      `<tr><td>\u{1F4C4} <a href="/${encodeURI(obj.key)}">${escapeHtml(name)}</a></td><td class="size">${obj.size.toLocaleString()} bytes</td></tr>`
    );
  }

  const nextLink =
    listed.truncated && listed.cursor
      ? `<p><a href="?cursor=${encodeURIComponent(listed.cursor)}">Next page &rarr;</a></p>`
      : "";

  const body = rows.length
    ? `<table><tbody>${rows.join("")}</tbody></table>${nextLink}`
    : `<p><em>(empty)</em></p>`;

  return new Response(
    `${PAGE_HEAD}<h2>Podcast Speech Dataset</h2>${renderCrumbs(prefix)}${body}` +
      `<p><a href="/v2/manifest/manifest.jsonl">Full manifest (JSONL, one row per clean clip)</a> &middot; ` +
      `<a href="https://github.com/rocketsri/podcast">repo / docs</a></p>${PAGE_FOOT}`,
    { status: 200, headers: { "content-type": "text/html; charset=utf-8" } }
  );
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const key = decodeURIComponent(url.pathname.replace(/^\/+/, ""));
    const isListing = key === "" || key.endsWith("/");

    if (request.method !== "GET" && request.method !== "HEAD") {
      return new Response("Method Not Allowed", { status: 405, headers: { Allow: "GET, HEAD" } });
    }

    if (isListing) {
      if (request.method === "HEAD") {
        return new Response(null, { status: 200, headers: { "content-type": "text/html; charset=utf-8" } });
      }
      return renderListing(env, key, request);
    }

    const object = await env.BUCKET.get(key, {
      onlyIf: request.headers,
      range: request.headers,
    });

    if (object === null) {
      return new Response("Object Not Found", { status: 404 });
    }

    const headers = new Headers();
    object.writeHttpMetadata(headers);
    headers.set("etag", object.httpEtag);
    headers.set("accept-ranges", "bytes");
    headers.set("access-control-allow-origin", "*");
    headers.set("cache-control", "public, max-age=31536000, immutable");

    if (object.range) {
      const end = "end" in object.range ? object.range.end : object.size - 1;
      headers.set("content-range", `bytes ${object.range.offset}-${end}/${object.size}`);
    }

    const status = object.body ? (request.headers.get("range") !== null ? 206 : 200) : 304;
    return new Response(object.body, { headers, status });
  },
};
