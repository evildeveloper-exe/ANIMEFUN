"""
AnimeSensei – Flask Server
==========================
Chatbot      → Groq API (llama-3.1-8b-instant, FREE tier)
Recommend    → 100% local ML (KNN, SVD, TF-IDF, GradBoost, Hybrid)

SETUP:
  1. Get FREE Groq key → https://console.groq.com
  2. Create .env file:   GROQ_API_KEY=gsk_xxxxxxxxxxxx
  3. pip install flask numpy scikit-learn joblib requests
  4. python model_trainer.py          ← train models (once)
  5. python local_server.py           ← start server
  6. Open → http://localhost:8080

Groq free tier: 14,400 req/day · 30 req/min
If no key → chatbot falls back to local NLP automatically.
"""

import os
import warnings
import requests
import numpy as np
from pathlib import Path
from datetime import date
from flask import Flask, request, jsonify, send_file
from sklearn.metrics.pairwise import cosine_similarity
import joblib

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────
GROQ_API_KEY = ""                       # paste key here OR use .env file
GROQ_MODEL   = "llama-3.1-8b-instant"  # fastest free Groq model
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
PORT         = 8080
MODELS_DIR   = Path("models")
DATA_FILE    = "anime_data.json"

app = Flask(__name__, static_folder=".")

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response

@app.route("/", methods=["OPTIONS"])
@app.route("/<path:path>", methods=["OPTIONS"])
def options_handler(path=""):
    from flask import Response
    r = Response()
    r.headers["Access-Control-Allow-Origin"]  = "*"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    r.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return r, 204


# ─────────────────────────────────────────────────────────
# LOAD .env  (no python-dotenv needed)
# ─────────────────────────────────────────────────────────
def load_dotenv():
    env = Path(".env")
    if not env.exists():
        return
    for raw in env.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))

load_dotenv()
if not GROQ_API_KEY:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


# ─────────────────────────────────────────────────────────
# LOAD TRAINED ML MODELS
# ─────────────────────────────────────────────────────────
def load(name):
    return joblib.load(MODELS_DIR / f"{name}.pkl")

print("[SERVER] Loading ML models …")
try:
    ANIME       = load("anime_list")
    FEAT_MAT    = load("feat_matrix")
    ALL_GENRES  = load("all_genres")
    ALL_MOODS   = load("all_moods")
    TFIDF_VEC   = load("tfidf_vec")
    TFIDF_MAT   = load("tfidf_matrix")
    KNN         = load("knn")
    LATENT_MAT  = load("latent_matrix")
    QUIZ_CLF    = load("quiz_clf")
    COLLAB      = load("collab")
    TOP_SIMILAR = load("top_similar")
    HYBRID_MAT  = load("hybrid_matrix")
    ID_IDX      = {a["id"]: i for i, a in enumerate(ANIME)}
    print(f"[SERVER] ✓  {len(ANIME)} anime · {FEAT_MAT.shape[1]} features")
except FileNotFoundError as e:
    print(f"[ERROR] {e}")
    print("[AUTO-TRAIN] Models not found — running model_trainer.py now...")
    import subprocess, sys
    subprocess.run([sys.executable, "model_trainer.py"], check=True)
    print("[AUTO-TRAIN] Training complete — reloading models...")
    ANIME       = load("anime_list")
    FEAT_MAT    = load("feat_matrix")
    ALL_GENRES  = load("all_genres")
    ALL_MOODS   = load("all_moods")
    TFIDF_VEC   = load("tfidf_vec")
    TFIDF_MAT   = load("tfidf_matrix")
    KNN         = load("knn")
    LATENT_MAT  = load("latent_matrix")
    QUIZ_CLF    = load("quiz_clf")
    COLLAB      = load("collab")
    TOP_SIMILAR = load("top_similar")
    HYBRID_MAT  = load("hybrid_matrix")

if GROQ_API_KEY:
    print(f"[SERVER] ✓  Groq chatbot → {GROQ_MODEL}")
else:
    print("[SERVER] ⚠  No Groq key – chatbot uses local NLP fallback")
    print("[SERVER]    Get a free key: https://console.groq.com")


# ─────────────────────────────────────────────────────────
# LOCAL ML RECOMMENDATION ENGINE
# ─────────────────────────────────────────────────────────
_MOOD_I  = {"dark": 0, "funny": 1, "emotional": 2, "relaxing": 3, "intense": 4}
_LEN_I   = {"1": 0, "short": 1, "medium": 2, "long": 3}
_INT_I   = {"intense": 0, "balanced": 1, "slow": 2, "action": 3}
_CHAR_I  = {"underdog": 0, "genius": 1, "op": 2, "complex": 3}
_POP_I   = {"popular": 0, "hidden": 1, "seasonal": 2, "any": 3}


def _quiz_vector(ans: dict) -> np.ndarray:
    gi = ALL_GENRES.index(ans.get("genre", ALL_GENRES[0])) if ans.get("genre") in ALL_GENRES else 0
    return np.array([[
        _MOOD_I.get(ans.get("mood", "dark"), 0) / 4.0,
        gi / max(len(ALL_GENRES) - 1, 1),
        _LEN_I.get(ans.get("length", "medium"), 1) / 3.0,
        _INT_I.get(ans.get("intensity", "balanced"), 1) / 3.0,
        _CHAR_I.get(ans.get("character", "underdog"), 0) / 3.0,
        _POP_I.get(ans.get("popularity", "any"), 3) / 3.0,
        0.85,
    ]], dtype=np.float32)


def recommend_quiz(ans: dict, n: int = 10) -> list:
    proba   = QUIZ_CLF.predict_proba(_quiz_vector(ans))[0]
    top5    = np.argsort(proba)[::-1][:5]
    ctrd    = FEAT_MAT[top5].mean(axis=0, keepdims=True)
    knn_set = set(KNN.kneighbors(ctrd)[1][0])
    hs      = HYBRID_MAT[top5].mean(axis=0)

    scored = []
    for i, a in enumerate(ANIME):
        s = proba[i] * 40 + hs[i] * 30 + (15 if i in knn_set else 0)
        for si in top5:
            s += COLLAB.get(ANIME[si]["id"], {}).get(a["id"], 0) * 10
        if ans.get("mood")  and ans["mood"]  in a.get("mood", []):    s += 20
        if ans.get("genre") and ans["genre"] in a.get("genres", []):  s += 20
        ep, lf = a.get("episodes", 1), ans.get("length", "")
        if lf == "1"      and ep == 1:            s += 15
        elif lf == "short"  and 2  <= ep <= 25:   s += 15
        elif lf == "medium" and 26 <= ep <= 75:   s += 15
        elif lf == "long"   and ep >= 76:          s += 15
        pop = ans.get("popularity", "any")
        if pop == "hidden"   and a.get("hidden_gem"): s += 12
        elif pop == "seasonal" and a.get("seasonal"): s += 12
        elif pop == "popular": s += a.get("popularity", 0) / 10
        s += a.get("rating", 0) * 2
        scored.append((s, a))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [dict(a, local_score=round(sc, 2)) for sc, a in scored[:n]]


def recommend_similar(anime_id: int, n: int = 8) -> list:
    idx = ID_IDX.get(anime_id)
    if idx is None:
        return []
    cscore = COLLAB.get(anime_id, {})
    scored = [(0.7 * e["score"] + 0.3 * cscore.get(e["id"], 0),
               next((x for x in ANIME if x["id"] == e["id"]), None))
              for e in TOP_SIMILAR.get(anime_id, [])]
    scored = [(s, a) for s, a in scored if a]
    if len(scored) < n:
        for ei in np.argsort(HYBRID_MAT[idx])[::-1]:
            if len(scored) >= n: break
            if ANIME[ei]["id"] == anime_id: continue
            if any(a["id"] == ANIME[ei]["id"] for _, a in scored): continue
            scored.append((HYBRID_MAT[idx][ei], ANIME[ei]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [a for _, a in scored[:n]]


def recommend_text(query: str, n: int = 10) -> list:
    sims  = cosine_similarity(TFIDF_VEC.transform([query.lower()]), TFIDF_MAT)[0]
    top_i = np.argsort(sims)[::-1][:n]
    return [dict(ANIME[i], text_score=round(float(sims[i]), 4)) for i in top_i]


def recommend_svd_picks(n: int = 10) -> list:
    scores  = np.array([a.get("rating", 0) * a.get("popularity", 0) for a in ANIME])
    weights = scores / scores.sum()
    ideal   = (LATENT_MAT * weights[:, np.newaxis]).sum(axis=0, keepdims=True)
    return [ANIME[i] for i in np.argsort(cosine_similarity(ideal, LATENT_MAT)[0])[::-1][:n]]


def search(query="", genre="", mood="", min_rating=0,
           year_b="", ep_b="", tab="") -> list:
    res = list(ANIME)
    if query:
        q    = query.lower()
        sims = cosine_similarity(TFIDF_VEC.transform([q]), TFIDF_MAT)[0]
        hits = [(sims[i], a) for i, a in enumerate(res)
                if sims[i] > 0.01 or q in a["title"].lower()
                or any(q in t.lower() for t in a.get("tags", []))]
        hits.sort(key=lambda x: x[0], reverse=True)
        res  = [a for _, a in hits] if hits else res
    if genre:      res = [a for a in res if genre in a.get("genres", [])]
    if mood:       res = [a for a in res if mood  in a.get("mood", [])]
    if min_rating: res = [a for a in res if a.get("rating", 0) >= min_rating]
    if year_b:
        y = int(year_b) if year_b.isdigit() else 0
        if   y == 2020: res = [a for a in res if a["year"] >= 2020]
        elif y == 2015: res = [a for a in res if 2015 <= a["year"] < 2020]
        elif y == 2010: res = [a for a in res if 2010 <= a["year"] < 2015]
        elif y == 2000: res = [a for a in res if 2000 <= a["year"] < 2010]
        elif y == 1990: res = [a for a in res if a["year"] < 2000]
    if ep_b:
        if   ep_b == "1":      res = [a for a in res if a["episodes"] == 1]
        elif ep_b == "short":  res = [a for a in res if 2  <= a["episodes"] <= 25]
        elif ep_b == "medium": res = [a for a in res if 26 <= a["episodes"] <= 75]
        elif ep_b == "long":   res = [a for a in res if a["episodes"] >= 76]
    if   tab == "trending":  res = [a for a in res if a.get("trending")]
    elif tab == "toprated":  res = sorted(res, key=lambda a: a.get("rating", 0), reverse=True)
    elif tab == "hidden":    res = [a for a in res if a.get("hidden_gem")]
    elif tab == "seasonal":  res = [a for a in res if a.get("seasonal")]
    elif tab == "ai-picks":  res = recommend_svd_picks(len(res))
    return res


# ─────────────────────────────────────────────────────────
# GROQ CHATBOT
# ─────────────────────────────────────────────────────────
def _db_context() -> str:
    """One-line summary per anime injected into Groq system prompt."""
    rows = []
    for a in ANIME:
        flags = ("gem " if a.get("hidden_gem") else "") + ("trending" if a.get("trending") else "")
        rows.append(
            f"• {a['title']} ({a['year']}) | ⭐{a['rating']} | "
            f"{a['episodes']} eps | {', '.join(a['genres'][:3])} | "
            f"mood:{','.join(a['mood'])} {flags}".rstrip()
        )
    return "\n".join(rows)


def _ml_context(message: str) -> str:
    """
    Run local ML on the message and return top matches as structured
    context — so Groq explains real results, never hallucinates titles.
    """
    msg = message.lower()
    mood_hits  = [m for m in ALL_MOODS  if m in msg]
    genre_hits = [g for g in ALL_GENRES if g.lower() in msg]
    length = ("1"     if any(w in msg for w in ["film", "movie", "one ep"]) else
              "short" if any(w in msg for w in ["short", "quick", "brief"]) else
              "long"  if any(w in msg for w in ["long", "epic", "hundreds"]) else "")
    hidden = any(w in msg for w in ["hidden", "underrated", "unknown", "gem"])

    ans = {
        "mood":       mood_hits[0]  if mood_hits  else "",
        "genre":      genre_hits[0] if genre_hits else "",
        "length":     length,
        "popularity": "hidden" if hidden else "popular",
    }

    tfidf = recommend_text(message, n=5)
    ml    = recommend_quiz(ans, n=5) if any(ans.values()) else []

    seen, merged = set(), []
    for a in ml + tfidf:
        if a["id"] not in seen:
            seen.add(a["id"])
            merged.append(a)

    if not merged:
        return ""

    lines = ["[ML pre-ranked matches — use these as primary recommendations:]"]
    for a in merged[:6]:
        lines.append(
            f"  {a['title']} | ⭐{a['rating']} | {a['episodes']} eps | "
            f"{', '.join(a['genres'][:2])} | {a['synopsis'][:90]}…"
        )
    return "\n".join(lines)


def groq_chat(message: str, history: list) -> tuple:
    """
    Call Groq API with:
      - Full anime DB as system context
      - ML pre-ranked picks injected so LLM explains real results
      - Multi-turn history (last 6 exchanges)
    Returns (reply_text, engine_label)
    Falls back to local_nlp() on any error.
    """
    if not GROQ_API_KEY:
        return local_nlp(message), "local-nlp"

    ml_ctx = _ml_context(message)

    system = f"""You are AnimeSensei AI, a friendly expert anime recommendation assistant.
You are enthusiastic, knowledgeable, use Japanese terms naturally (nakama, senpai, sugoi).

ANIME DATABASE ({len(ANIME)} titles — ONLY recommend from this list, never invent):
{_db_context()}

RULES:
- Always include: title, rating (⭐), episode count, genre, and WHY it fits the user.
- For comparisons give a clear verdict with specific reasons.
- Keep responses under 350 words. Use emojis for readability.
- If asked something unrelated to anime, gently redirect.

{ml_ctx}"""

    msgs = [{"role": "system", "content": system}]
    msgs.extend(history[-12:])          # last 6 turns (12 messages)
    msgs.append({"role": "user", "content": message})

    try:
        r = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type":  "application/json",
            },
            json={
                "model":       GROQ_MODEL,
                "messages":    msgs,
                "max_tokens":  600,
                "temperature": 0.75,
                "top_p":       0.9,
            },
            timeout=15,
        )

        if r.status_code == 200:
            reply = r.json()["choices"][0]["message"]["content"].strip()
            return reply, f"groq/{GROQ_MODEL}"

        elif r.status_code == 429:
            print("[GROQ] Rate limited → local NLP fallback")
            reply = local_nlp(message) + "\n\n_⚡ Groq rate limit hit — used local ML_"
            return reply, "local-nlp (rate limited)"

        elif r.status_code == 401:
            print("[GROQ] Invalid API key")
            reply = ("⚠️ **Invalid Groq API key.**\n"
                     "Check your `.env` file or `GROQ_API_KEY` setting.\n\n"
                     + local_nlp(message))
            return reply, "local-nlp (bad key)"

        else:
            print(f"[GROQ] HTTP {r.status_code}: {r.text[:120]}")
            return local_nlp(message), "local-nlp (error)"

    except requests.exceptions.Timeout:
        print("[GROQ] Timeout → local NLP fallback")
        return local_nlp(message) + "\n\n_⏱ Groq timed out — used local ML_", "local-nlp (timeout)"

    except Exception as e:
        print(f"[GROQ] Exception: {e}")
        return local_nlp(message), "local-nlp (exception)"


# ─────────────────────────────────────────────────────────
# LOCAL NLP FALLBACK  (no Groq key / rate limit / offline)
# ─────────────────────────────────────────────────────────
def local_nlp(message: str) -> str:
    msg = message.lower().strip()

    # Intent: compare two anime
    if any(kw in msg for kw in ["compare", " vs ", "versus", "better between"]):
        matched = [a for a in ANIME if a["title"].lower() in msg]
        if len(matched) >= 2:
            a, b = matched[0], matched[1]
            w = a if a["rating"] >= b["rating"] else b
            l = b if w == a else a
            return (
                f"⚔️ **{a['title']}** (⭐{a['rating']}, {a['episodes']} eps)  vs  "
                f"**{b['title']}** (⭐{b['rating']}, {b['episodes']} eps)\n\n"
                f"**{w['title']}** wins on rating (⭐{w['rating']}).\n"
                f"Pick **{w['title']}** for {', '.join(w['genres'][:2])}; "
                f"**{l['title']}** if you prefer {', '.join(l['genres'][:2])}."
            )

    # Intent: info about a specific anime
    for a in ANIME:
        if a["title"].lower() in msg:
            sims = ", ".join(f"**{s['title']}**" for s in recommend_similar(a["id"], n=3))
            return (
                f"**{a['title']}** — ⭐{a['rating']} | {a['episodes']} eps | "
                f"{', '.join(a['genres'][:3])}\n\n"
                f"📖 {a['synopsis'][:200]}…\n\n"
                f"Also try: {sims}"
            )

    # Intent: general recommendation
    mood_hits  = [m for m in ALL_MOODS  if m in msg]
    genre_hits = [g for g in ALL_GENRES if g.lower() in msg]
    length = ("1"     if any(w in msg for w in ["film", "movie"]) else
              "short" if "short" in msg else
              "long"  if "long"  in msg else "")

    ans = {
        "mood":       mood_hits[0]  if mood_hits  else "",
        "genre":      genre_hits[0] if genre_hits else "",
        "length":     length,
        "popularity": "hidden" if any(w in msg for w in ["hidden", "underrated"]) else "popular",
    }
    recs = recommend_quiz(ans, n=5) if any(ans.values()) else recommend_text(message, n=5)

    if not recs:
        return ("I couldn't find a specific match. Try asking for a genre or mood — "
                "like *'dark psychological anime'* or *'funny short series'*.")

    lines = ["🎌 Here are your ML-matched picks:\n"]
    for r in recs[:5]:
        lines.append(
            f"**{r['title']}** (⭐{r['rating']}, {r['episodes']} eps) — "
            f"{', '.join(r['genres'][:2])}\n{r['synopsis'][:100]}…\n"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────
# FLASK API ROUTES
# ─────────────────────────────────────────────────────────

@app.route("/")
@app.route("/index.html")
def serve_frontend():
    if Path("index.html").exists():
        return send_file("index.html")
    return "Place index.html in the same folder.", 404


@app.route("/anime_data.json")
def serve_data():
    return send_file(DATA_FILE)


# ── Anime endpoints ────────────────────────────────────────

@app.route("/api/anime")
def api_anime():
    return jsonify({"count": len(ANIME),
                    "anime": search(tab=request.args.get("tab", ""))})


@app.route("/api/anime/search")
def api_search():
    res = search(
        query      = request.args.get("q", ""),
        genre      = request.args.get("genre", ""),
        mood       = request.args.get("mood", ""),
        min_rating = float(request.args.get("rating", 0) or 0),
        year_b     = request.args.get("year", ""),
        ep_b       = request.args.get("eps", ""),
        tab        = request.args.get("tab", ""),
    )
    return jsonify({"count": len(res), "anime": res})


@app.route("/api/anime/daily")
def api_daily():
    return jsonify(ANIME[date.today().toordinal() % len(ANIME)])


@app.route("/api/anime/stats")
def api_stats():
    ratings = [a["rating"] for a in ANIME]
    return jsonify({
        "total":          len(ANIME),
        "avg_rating":     round(sum(ratings) / len(ratings), 2),
        "genres":         ALL_GENRES,
        "moods":          ALL_MOODS,
        "trending_count": sum(1 for a in ANIME if a.get("trending")),
        "hidden_gems":    sum(1 for a in ANIME if a.get("hidden_gem")),
        "seasonal_count": sum(1 for a in ANIME if a.get("seasonal")),
        "chatbot_engine": f"groq/{GROQ_MODEL}" if GROQ_API_KEY else "local-nlp",
        "groq_key_set":   bool(GROQ_API_KEY),
    })


@app.route("/api/anime/<int:anime_id>")
def api_anime_detail(anime_id):
    a = next((x for x in ANIME if x["id"] == anime_id), None)
    return jsonify(a) if a else (jsonify({"error": "Not found"}), 404)


@app.route("/api/anime/<int:anime_id>/similar")
def api_similar(anime_id):
    n = int(request.args.get("n", 6))
    return jsonify({"anime_id": anime_id,
                    "similar": recommend_similar(anime_id, n=n)})


@app.route("/api/compare/<int:id_a>/<int:id_b>")
def api_compare(id_a, id_b):
    a = next((x for x in ANIME if x["id"] == id_a), None)
    b = next((x for x in ANIME if x["id"] == id_b), None)
    if not a or not b:
        return jsonify({"error": "Anime not found"}), 404
    hy     = float(HYBRID_MAT[ID_IDX[id_a]][ID_IDX[id_b]])
    wr     = "a" if a["rating"]     >= b["rating"]     else "b"
    wp     = "a" if a["popularity"] >= b["popularity"] else "b"
    eps    = lambda e: 1000 if e == 1 else max(0, 100 - min(e, 100))
    wc     = "a" if eps(a["episodes"]) >= eps(b["episodes"]) else "b"
    aw     = sum([wr == "a", wp == "a", wc == "a"])
    return jsonify({
        "anime_a": a, "anime_b": b, "similarity": round(hy, 3),
        "winner_rating": wr, "winner_pop": wp, "winner_compact": wc,
        "overall_winner": "a" if aw >= 2 else "b",
    })


# ── Recommendation endpoints ───────────────────────────────

@app.route("/api/recommend", methods=["GET", "POST"])
def api_recommend():
    if request.method == "POST":
        body = request.get_json(force=True) or {}
        ans  = body.get("answers", body)
        n    = int(body.get("n", 10))
    else:
        ans = {k: request.args.get(k, "")
               for k in ["mood", "genre", "length", "intensity", "character", "popularity"]}
        n   = int(request.args.get("n", 10))
    recs = recommend_quiz(ans, n=n)
    return jsonify({"count": len(recs), "recommendations": recs, "engine": "local-ml"})


@app.route("/api/recommend/text")
def api_rec_text():
    q = request.args.get("q", "")
    if not q:
        return jsonify({"error": "Provide ?q="}), 400
    n = int(request.args.get("n", 10))
    return jsonify({"recommendations": recommend_text(q, n), "engine": "tfidf"})


@app.route("/api/recommend/ai-picks")
def api_ai_picks():
    n = int(request.args.get("n", 10))
    return jsonify({"recommendations": recommend_svd_picks(n), "engine": "svd-local"})


# ── Groq Chatbot ───────────────────────────────────────────
_sessions: dict = {}   # { session_id → [{"role": .., "content": ..}, …] }


@app.route("/api/chat", methods=["POST"])
def api_chat():
    body    = request.get_json(force=True) or {}
    message = body.get("message", "").strip()
    session = body.get("session", "default")

    if not message:
        return jsonify({"error": "Empty message"}), 400

    history = _sessions.setdefault(session, [])
    reply, engine = groq_chat(message, history)

    # Persist this turn
    history.append({"role": "user",      "content": message})
    history.append({"role": "assistant", "content": reply})
    if len(history) > 24:                 # keep last 12 turns
        _sessions[session] = history[-24:]

    return jsonify({"reply": reply, "engine": engine, "session": session})


@app.route("/api/chat/clear", methods=["POST"])
def api_chat_clear():
    session = (request.get_json(force=True) or {}).get("session", "default")
    _sessions.pop(session, None)
    return jsonify({"cleared": True, "session": session})


@app.route("/api/status")
def api_status():
    return jsonify({
        "groq_key_set":   bool(GROQ_API_KEY),
        "groq_model":     GROQ_MODEL if GROQ_API_KEY else None,
        "chatbot_engine": f"groq/{GROQ_MODEL}" if GROQ_API_KEY else "local-nlp",
        "anime_count":    len(ANIME),
        "ml_models":      ["tfidf", "knn", "svd", "gradient_boosting", "collab", "hybrid"],
    })


# ── Watchlist (in-memory per session) ─────────────────────
_watchlists: dict = {}


@app.route("/api/watchlist")
def api_watchlist():
    sid  = request.args.get("session", "default")
    ids  = _watchlists.get(sid, [])
    return jsonify({"session": sid, "count": len(ids),
                    "watchlist": [a for a in ANIME if a["id"] in ids]})


@app.route("/api/watchlist/toggle", methods=["POST"])
def api_watchlist_toggle():
    body = request.get_json(force=True) or {}
    sid  = body.get("session", "default")
    aid  = int(body.get("anime_id", 0))
    lst  = _watchlists.setdefault(sid, [])
    if aid in lst:
        lst.remove(aid); added = False
    else:
        lst.append(aid); added = True
    return jsonify({"added": added, "watchlist": lst})


# ── Retrain ────────────────────────────────────────────────
@app.route("/api/retrain")
def api_retrain():
    import subprocess, sys
    r = subprocess.run([sys.executable, "model_trainer.py"],
                       capture_output=True, text=True, timeout=120)
    if r.returncode == 0:
        return jsonify({"status": "ok", "message": "Retrained. Restart server to apply."})
    return jsonify({"status": "error", "output": r.stderr}), 500


# ─────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────
def banner():
    ks = f"✓ Groq ({GROQ_MODEL})" if GROQ_API_KEY else "✗ Not set  (add to .env)"
    print(f"""
╔══════════════════════════════════════════════════════╗
║  ⛩  AnimeSensei  –  ML Server  +  Groq Chatbot     ║
╠══════════════════════════════════════════════════════╣
║  Open   →  http://localhost:{PORT}                     ║
║  Status →  http://localhost:{PORT}/api/status          ║
║                                                      ║
║  Chatbot  : {ks:<42}║
║  ML Recs  : Local · KNN · SVD · TF-IDF · Hybrid     ║
╠══════════════════════════════════════════════════════╣
║  POST /api/chat              Groq AI chatbot         ║
║  POST /api/chat/clear        Clear chat history      ║
║  POST /api/recommend         Quiz ML recommendations ║
║  GET  /api/anime/search      TF-IDF search           ║
║  GET  /api/anime/<id>/similar Hybrid ML similar      ║
║  GET  /api/status            Groq key / engine info  ║
╚══════════════════════════════════════════════════════╝
  Press Ctrl+C to stop.
""")


if __name__ == "__main__":
    banner()
    app.run(host="0.0.0.0", port=PORT, debug=False)
