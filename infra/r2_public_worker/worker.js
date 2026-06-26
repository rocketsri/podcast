// Public read-only proxy for the `podcast` R2 bucket (dataset deliverable).
// Deployed via `wrangler deploy` -- see README in this directory for the exact command.
// Pattern follows Cloudflare's own R2-public-read Worker example (same shape as
// https://denoflare.dev/examples/r2-public-read): no auth, GET/HEAD only,
// range + conditional request support so audio clients can byte-range FLAC clips.

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const key = decodeURIComponent(url.pathname.replace(/^\/+/, ""));

    if (request.method !== "GET" && request.method !== "HEAD") {
      return new Response("Method Not Allowed", { status: 405, headers: { Allow: "GET, HEAD" } });
    }

    if (key === "") {
      return new Response(
        "Podcast Speech Dataset -- public read-only mirror of the `podcast` R2 bucket.\n" +
          "Manifest: /v2/manifest/manifest.jsonl\n" +
          "Clips:    /v2/clips/<podcast_id>/<episode_id>/<clip_id>.flac\n" +
          "See https://github.com/rocketsri/podcast for full docs.\n",
        { status: 200, headers: { "content-type": "text/plain; charset=utf-8" } }
      );
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
