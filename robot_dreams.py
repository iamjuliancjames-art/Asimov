# =============================================================================
#  Martian v0.3 — improved bolt-on manifold memory (M4 + Talos inspired)
#  Pure stdlib Python 3.11+
#
#  Fixes vs your v2:
#   - Stability is no longer always 1.0 (sigmoid mapping; real discrimination)
#   - Novelty is "distance from recent memory" (not char-entropy)
#   - Pruning is value-based, anchor-protected, and prefers pruning archived first
#   - Non-destructive consolidation: reflect() builds semantic hubs with provenance edges
#   - Graph is used in retrieval (1-hop expansion + proximity bonus)
#   - Lotus edge cost uses BOTH endpoint properties (pi/risk)
# =============================================================================

import math
import time
import re
import random
from collections import defaultdict, deque, Counter
from typing import List, Dict, Optional, Tuple, Set


class Martian:
    """
    Advanced in-memory manifold memory system.
    - Rooms with scored metadata (importance/novelty/nuance/stability/kind)
    - Small-world-ish graph linking similar rooms with Lotus-weighted edge costs
    - Retrieval uses both text similarity and graph proximity expansion
    - Talos drift monitoring + attractor nudge
    - Identity anchors (high-stability continuity)
    - Non-destructive consolidation (reflect): episodic -> semantic hub w/ provenance edges
    - Value-based pruning with anchor protection + archive-first eviction
    """

    def __init__(self,
                 max_rooms: int = 800,
                 sim_threshold: float = 0.25,
                 history_window: int = 40,
                 reflect_min_cluster: int = 6,
                 reflect_max_sources: int = 24,
                 reflect_recent_hours: float = 72.0):
        self.rooms: List[Dict] = []
        self.room_id_counter = 0
        self.max_rooms = max_rooms
        self.sim_threshold = sim_threshold

        # Graph: room_id → {neighbor_id: lotus_cost}
        self.graph: Dict[int, Dict[int, float]] = defaultdict(dict)

        # Access order for LRU-ish weighting (ids)
        self.access_order = deque(maxlen=max_rooms * 2)

        # Identity anchors (ids)
        self.anchor_ids: Set[int] = set()

        # Talos state
        self.recent_texts = deque(maxlen=history_window)
        self.attractors: List[str] = []

        # Reflect settings
        self.reflect_min_cluster = reflect_min_cluster
        self.reflect_max_sources = reflect_max_sources
        self.reflect_recent_hours = reflect_recent_hours

        # Constants / weights (tuneable)
        self.EPS = 1e-10

        # Lotus cost parameters (conceptual knobs)
        self.LAMBDA_PI = 0.30
        self.MU_RISK = 0.60
        self.SINGULARITY_RISK_GATE = 0.80
        self.SINGULARITY_PENALTY = 1.00  # multiplier for divergence penalty

        # Retrieval weights
        self.KIND_PRIORITY = {
            "semantic": 1.00,
            "state": 0.90,
            "commitment": 0.85,
            "episodic": 0.55,
            "unknown": 0.30
        }
        self.W_SIM = 0.45
        self.W_KIND = 0.18
        self.W_IMPORTANCE = 0.12
        self.W_STABILITY = 0.10
        self.W_RECENCY = 0.10
        self.W_GRAPH = 0.05

        # Pruning weights
        self.PRUNE_PROTECT_ANCHORS = True
        self.PRUNE_KEEP_SEMANTIC_BIAS = 0.15  # semantic rooms get extra value
        self.PRUNE_ARCHIVE_FIRST = True

    # ──────────────────────────────────────────────────────────────────────────
    # Similarity (stdlib stand-in for embedding cosine)
    # ──────────────────────────────────────────────────────────────────────────
    def _pseudo_sim(self, a: str, b: str) -> float:
        """Crude n-gram + length-aware similarity (stand-in for cosine)."""
        if not a or not b:
            return 0.0
        a, b = a.lower(), b.lower()

        def ngrams(s: str, n: int) -> Set[str]:
            if len(s) < n:
                return set()
            return {s[i:i+n] for i in range(len(s)-n+1)}

        a3, b3 = ngrams(a, 3), ngrams(b, 3)
        a4, b4 = ngrams(a, 4), ngrams(b, 4)

        def jacc(x, y):
            if not x and not y:
                return 0.0
            return len(x & y) / max(1, len(x | y))

        ov = max(jacc(a3, b3), jacc(a4, b4), 0.0)
        len_r = min(len(a), len(b)) / max(1, max(len(a), len(b)))
        return ov * (0.35 + 0.65 * (len_r ** 1.25))

    # ──────────────────────────────────────────────────────────────────────────
    # Feature extraction / scoring
    # ──────────────────────────────────────────────────────────────────────────
    def _nuance(self, text: str) -> float:
        """Unique-token ratio proxy for nuance."""
        words = re.findall(r'[a-z0-9]+', text.lower())
        if not words:
            return 0.0
        return len(set(words)) / len(words)

    def _char_entropy(self, text: str) -> float:
        """Character Shannon entropy proxy for 'texture' (not semantic novelty)."""
        if not text:
            return 0.0
        t = text.lower()
        counts = Counter(t)
        total = len(t)
        h = 0.0
        for c, k in counts.items():
            p = k / total
            h -= p * math.log2(p + self.EPS)
        return h

    def _sigmoid(self, x: float) -> float:
        # numerically stable-ish sigmoid
        if x >= 0:
            z = math.exp(-x)
            return 1 / (1 + z)
        z = math.exp(x)
        return z / (1 + z)

    def _novelty_vs_recent(self, text: str, lookback: int = 40) -> float:
        """
        Novelty is 1 - max similarity to recent rooms (or 1 if empty).
        In [0,1]. Higher means more novel.
        """
        if not self.rooms:
            return 1.0
        # use only last N rooms for novelty baseline (cheap)
        recent = self.rooms[-min(len(self.rooms), lookback):]
        max_sim = 0.0
        for r in recent:
            if r["meta"].get("archived"):
                continue
            s = self._pseudo_sim(text, r["text"])
            if s > max_sim:
                max_sim = s
        return max(0.0, min(1.0, 1.0 - max_sim))

    def _stability(self, novelty: float, nuance: float, kind: str) -> float:
        """
        Stability ∈ (0,1]. Higher = more durable.
        Uses a sigmoid mapping so it's not always 1.0.
        Adds kind prior: semantic/state/commitment tend to be more stable.
        """
        kind_bias = {
            "semantic": 0.45,
            "state": 0.25,
            "commitment": 0.30,
            "episodic": -0.05,
            "unknown": 0.00
        }.get(kind, 0.00)

        # novelty adds stability a bit (novel things are worth keeping), nuance adds more
        # shift so average text lands mid-range
        x = (-0.6 + 1.2 * novelty + 1.8 * nuance + kind_bias)
        s = self._sigmoid(x)  # (0,1)
        return max(0.05, min(1.0, s))

    def _importance(self, text: str, ts: float, novelty: float) -> float:
        now = time.time()
        age_h = (now - ts) / 3600.0
        recency = max(0.05, 1.0 / (1.0 + age_h * 0.12))  # decays with hours

        # length proxy saturates around 120 words
        length_term = min(1.0, len(text.split()) / 120.0)

        # novelty_term saturates around 0.8 novelty
        novelty_term = min(1.0, novelty / 0.8)

        # importance ∈ [0,1]
        imp = 0.45 * recency + 0.30 * length_term + 0.25 * novelty_term
        return max(0.02, min(1.0, imp))

    def _lotus_cost(self, dist: float, pi_a: float, pi_b: float, risk_a: float, risk_b: float) -> float:
        """
        Lotus-weighted edge cost.
        - dist is base distance
        - pi term: average endpoint "phase index"
        - risk term: max endpoint risk
        - singularity divergence penalty when risk is high
        """
        pi = 0.5 * (pi_a + pi_b)
        risk = max(risk_a, risk_b)

        pi_term = self.LAMBDA_PI * pi
        risk_term = self.MU_RISK * risk

        sing = 0.0
        if risk > self.SINGULARITY_RISK_GATE:
            # blow up as risk→1; bounded by eps
            sing = self.SINGULARITY_PENALTY * (1.0 / max(self.EPS, (1.0 - risk)))

        return dist + pi_term + risk_term + sing

    def _room_by_id(self, rid: int) -> Optional[Dict]:
        # linear lookup is fine at <=800; keep it simple
        for r in self.rooms:
            if r["id"] == rid:
                return r
        return None

    def _score_room_meta(self, text: str, ts: float, kind: str, pi: Optional[float] = None, risk: Optional[float] = None) -> Dict:
        novelty = self._novelty_vs_recent(text)
        nuance_val = self._nuance(text)
        texture = self._char_entropy(text)

        stability = self._stability(novelty, nuance_val, kind)
        importance = self._importance(text, ts, novelty)

        # pseudo pi/risk defaults: deterministic-ish noise
        now = time.time()
        pi_val = float(pi) if pi is not None else round((now % 1.0), 4)
        risk_val = float(risk) if risk is not None else round(((now % 1000.0) / 1000.0) * 0.6, 4)

        return {
            "kind": kind,
            "kind_priority": self.KIND_PRIORITY.get(kind, 0.30),
            "ts": ts,
            "novelty": round(novelty, 4),
            "nuance": round(nuance_val, 4),
            "texture_entropy": round(texture, 4),
            "stability": round(stability, 4),
            "importance": round(importance, 4),
            "pi": pi_val,
            "risk": risk_val,
            "archived": False,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Core operations
    # ──────────────────────────────────────────────────────────────────────────
    def add(self,
            text: str,
            kind: str = "unknown",
            metadata: Optional[Dict] = None,
            is_anchor: bool = False,
            attractor: bool = False,
            pi: Optional[float] = None,
            risk: Optional[float] = None) -> int:
        if not text or not text.strip():
            return -1

        # near-duplicate skip (active rooms only)
        max_sim = 0.0
        for r in self.rooms:
            if r["meta"].get("archived"):
                continue
            max_sim = max(max_sim, self._pseudo_sim(text, r["text"]))
        if max_sim > 0.95:
            return -1

        rid = self.room_id_counter
        self.room_id_counter += 1
        ts = time.time()

        meta = self._score_room_meta(text, ts, kind, pi=pi, risk=risk)

        room = {
            "id": rid,
            "text": text,
            "meta": meta,
            "links": {
                # provenance: semantic hub -> source ids
                "sources": [],
                # membership: source -> semantic hubs
                "hubs": [],
            }
        }
        if metadata:
            # allow user to attach extra fields
            room.update(metadata)

        self.rooms.append(room)
        self.access_order.append(rid)

        if is_anchor:
            self.anchor_ids.add(rid)

        if attractor:
            self.attractors.append(text)

        # connect in graph to top neighbors
        self._connect_room(rid)

        # enforce capacity (prefer pruning archived first)
        self._enforce_capacity()

        return rid

    def _connect_room(self, rid: int):
        r = self._room_by_id(rid)
        if not r:
            return
        text = r["text"]
        meta = r["meta"]

        sims: List[Tuple[float, int]] = []
        for other in self.rooms:
            oid = other["id"]
            if oid == rid:
                continue
            if other["meta"].get("archived"):
                continue
            s = self._pseudo_sim(text, other["text"])
            sims.append((s, oid))
        sims.sort(reverse=True)

        # small-world-ish: top 7 similar
        for sim_val, oid in sims[:7]:
            if sim_val < self.sim_threshold:
                continue
            o = self._room_by_id(oid)
            if not o:
                continue
            dist = 1.0 - sim_val
            cost = self._lotus_cost(dist, meta["pi"], o["meta"]["pi"], meta["risk"], o["meta"]["risk"])
            cost = round(cost, 6)
            self.graph[rid][oid] = cost
            self.graph[oid][rid] = cost

    def retrieve(self, query: str, top_k: int = 6, min_sim: float = 0.20, expand_hops: int = 1) -> List[Dict]:
        """
        Retrieval:
         1) base scoring by similarity + meta factors
         2) expand by graph neighbors (1 hop by default) with proximity bonus
        """
        if not self.rooms or not query:
            return []

        now = time.time()
        base_scores: Dict[int, float] = {}

        # Base pass
        for room in self.rooms:
            rid = room["id"]
            if room["meta"].get("archived"):
                continue
            sim = self._pseudo_sim(query, room["text"])
            if sim < min_sim:
                continue

            age_days = (now - room["meta"]["ts"]) / 86400.0
            recency = 1.0 / (1.0 + age_days)

            kind_pri = room["meta"]["kind_priority"]
            imp = room["meta"]["importance"]
            stab = room["meta"]["stability"]

            score = (
                self.W_SIM * sim +
                self.W_KIND * kind_pri +
                self.W_IMPORTANCE * imp +
                self.W_STABILITY * stab +
                self.W_RECENCY * recency
            )

            # anchors get a small bonus if they match at all
            if rid in self.anchor_ids:
                score += 0.05

            base_scores[rid] = score

        if not base_scores:
            return []

        # Graph expansion: neighbors of strong base hits get a smaller derived score
        expanded_scores = dict(base_scores)
        if expand_hops >= 1:
            # take a few seeds
            seeds = sorted(base_scores.items(), key=lambda x: x[1], reverse=True)[:max(6, top_k)]
            for seed_id, seed_score in seeds:
                for nb, cost in self.graph.get(seed_id, {}).items():
                    nb_room = self._room_by_id(nb)
                    if not nb_room or nb_room["meta"].get("archived"):
                        continue
                    # lower cost → higher proximity
                    proximity = 1.0 / (1.0 + cost)
                    bonus = self.W_GRAPH * proximity * (0.6 + 0.4 * seed_score)
                    expanded_scores[nb] = max(expanded_scores.get(nb, 0.0), seed_score * 0.35 + bonus)

        ranked = sorted(expanded_scores.items(), key=lambda x: x[1], reverse=True)
        out = []
        for rid, _ in ranked[:top_k]:
            r = self._room_by_id(rid)
            if r:
                out.append(r)
                self.access_order.append(rid)
        return out

    def get_context(self, query: str, max_chars: int = 2400) -> str:
        parts = []

        # anchors (continuity)
        if self.anchor_ids:
            parts.append("Identity anchors:")
            # show last 2 anchors by time
            anchors = [self._room_by_id(aid) for aid in self.anchor_ids]
            anchors = [a for a in anchors if a and not a["meta"].get("archived")]
            anchors.sort(key=lambda r: r["meta"]["ts"])
            for a in anchors[-2:]:
                parts.append(f"• {a['text'][:160]}…")

        # attractors (goals)
        if self.attractors:
            parts.append("\nCore attractors to preserve:")
            for att in self.attractors[-3:]:
                parts.append(f"• {att[:160]}")

        relevant = self.retrieve(query, top_k=8, min_sim=0.18, expand_hops=1)
        if relevant:
            parts.append("\nRelevant rooms:")
            now = time.time()
            for r in relevant:
                age_h = int((now - r["meta"]["ts"]) / 3600.0)
                s = r["meta"]
                line = (
                    f"[{age_h}h | {s['kind']}] {r['text'][:180]}… "
                    f"(stab:{s['stability']:.2f} imp:{s['importance']:.2f} nov:{s['novelty']:.2f})"
                )
                parts.append(line)

        ctx = "\n".join(parts)
        return (ctx[:max_chars] + "…") if len(ctx) > max_chars else ctx

    # ──────────────────────────────────────────────────────────────────────────
    # Consolidation (Reflect) — non-destructive semantic hubs with provenance
    # ──────────────────────────────────────────────────────────────────────────
    def reflect(self) -> Optional[int]:
        """
        Consolidate recent episodic/unknown rooms into a semantic hub.
        - Finds a dense-ish cluster among recent low-stability rooms
        - Creates a semantic summary hub with links to sources
        - Archives (does not delete) the sources (optional behavior)
        Returns hub_id or None.
        """
        if len(self.rooms) < (self.reflect_min_cluster + 2):
            return None

        now = time.time()
        horizon = self.reflect_recent_hours * 3600.0

        # candidates: recent, not archived, not anchors, lowish stability, episodic-ish
        candidates = []
        for r in self.rooms:
            if r["meta"].get("archived"):
                continue
            rid = r["id"]
            if self.PRUNE_PROTECT_ANCHORS and rid in self.anchor_ids:
                continue
            if (now - r["meta"]["ts"]) > horizon:
                continue
            if r["meta"]["kind"] not in ("episodic", "unknown", "state"):
                continue
            if r["meta"]["stability"] > 0.62:
                continue
            candidates.append(r)

        if len(candidates) < self.reflect_min_cluster:
            return None

        # build a cluster around the most "central" candidate (max avg sim to others)
        # O(n^2) with n small (recent window). Fine.
        best_center = None
        best_center_score = -1.0
        for r in candidates:
            sims = []
            for o in candidates:
                if o["id"] == r["id"]:
                    continue
                sims.append(self._pseudo_sim(r["text"], o["text"]))
            if not sims:
                continue
            score = sum(sorted(sims, reverse=True)[:min(8, len(sims))]) / max(1, min(8, len(sims)))
            if score > best_center_score:
                best_center_score = score
                best_center = r

        if not best_center:
            return None

        # pick members: those above similarity threshold to center
        members = []
        center_text = best_center["text"]
        for r in candidates:
            s = self._pseudo_sim(center_text, r["text"])
            if s >= max(self.sim_threshold, 0.28):
                members.append((s, r))
        members.sort(reverse=True, key=lambda x: x[0])
        members = [r for _, r in members[:self.reflect_max_sources]]

        if len(members) < self.reflect_min_cluster:
            return None

        # generate a compact semantic summary (stdlib heuristic)
        hub_text = self._summarize_cluster(members)

        hub_id = self.add(
            hub_text,
            kind="semantic",
            metadata={"links": {"sources": [m["id"] for m in members], "hubs": []}},
        )
        hub = self._room_by_id(hub_id)
        if not hub:
            return None

        # update provenance: each member points to hub
        for m in members:
            m_room = self._room_by_id(m["id"])
            if not m_room:
                continue
            m_room["links"].setdefault("hubs", [])
            if hub_id not in m_room["links"]["hubs"]:
                m_room["links"]["hubs"].append(hub_id)

            # connect hub strongly to its sources (low dist)
            m_meta = m_room["meta"]
            h_meta = hub["meta"]
            # set sim to be fairly high because it's a consolidation relation
            dist = 0.20
            cost = self._lotus_cost(dist, h_meta["pi"], m_meta["pi"], h_meta["risk"], m_meta["risk"])
            self.graph[hub_id][m_room["id"]] = round(cost, 6)
            self.graph[m_room["id"]][hub_id] = round(cost, 6)

            # archive sources (soft), not delete
            m_room["meta"]["archived"] = True

        self._enforce_capacity()
        return hub_id

    def _summarize_cluster(self, members: List[Dict]) -> str:
        """
        Stdlib summarizer:
         - pull top keywords across member texts
         - stitch 2–4 short exemplars
        """
        texts = [m["text"] for m in members]
        words = []
        for t in texts:
            words += re.findall(r"[a-z0-9']+", t.lower())
        stop = {
            "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is", "are", "was", "were",
            "it", "this", "that", "i", "you", "we", "they", "he", "she", "as", "at", "by", "from", "be", "been",
            "not", "no", "but", "so", "if", "then", "than", "into", "about"
        }
        words = [w for w in words if w not in stop and len(w) >= 3]
        cnt = Counter(words)
        top = [w for w, _ in cnt.most_common(8)]
        topic = ", ".join(top[:6]) if top else "mixed themes"

        # pick a few short exemplars (first 30–60 chars)
        exemplars = []
        for m in members[:4]:
            s = m["text"].strip().replace("\n", " ")
            exemplars.append(s[:60] + ("…" if len(s) > 60 else ""))

        return (
            f"Consolidated semantic hub ({len(members)} sources). "
            f"Topic keywords: {topic}. "
            f"Exemplars: " + " / ".join(exemplars)
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Talos control features
    # ──────────────────────────────────────────────────────────────────────────
    def talos_check(self, new_text: str) -> Dict:
        """
        Drift monitor:
        - Entropy over recent token distribution (topic collapse vs diversity)
        - Coherence proxy: immediate repetition rate
        """
        if not new_text:
            return {"stable": True, "nudge_suggestion": None}

        self.recent_texts.append(new_text.lower())
        if len(self.recent_texts) < 5:
            return {"stable": True, "nudge_suggestion": None}

        words = []
        for t in self.recent_texts:
            words.extend(re.findall(r"[a-z0-9']+", t))

        if not words:
            return {"stable": True, "nudge_suggestion": None}

        cnt = Counter(words)
        total = len(words)
        ent = -sum((c/total) * math.log2((c/total) + self.EPS) for c in cnt.values())

        repeats = sum(1 for i in range(1, len(words)) if words[i] == words[i-1])
        coherence = 1.0 - min(0.95, repeats / max(1, (total - 1)))

        # drift if entropy too low (topic lock) or coherence too low (stutter loop)
        drift = (ent < 2.6) or (coherence < 0.50)

        nudge = None
        if drift and self.attractors:
            nudge = f"Pull toward attractor: {self.attractors[-1][:120]}…"

        return {
            "stable": not drift,
            "entropy": round(ent, 3),
            "coherence_proxy": round(coherence, 3),
            "nudge_suggestion": nudge
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Capacity management / pruning
    # ──────────────────────────────────────────────────────────────────────────
    def _room_value(self, room: Dict) -> float:
        """
        Value score used for pruning.
        Higher value = keep.
        """
        now = time.time()
        age_days = (now - room["meta"]["ts"]) / 86400.0
        recency = 1.0 / (1.0 + age_days)

        kind = room["meta"]["kind"]
        kind_pri = room["meta"]["kind_priority"]
        imp = room["meta"]["importance"]
        stab = room["meta"]["stability"]

        v = (0.35 * imp) + (0.25 * stab) + (0.20 * kind_pri) + (0.20 * recency)

        if kind == "semantic":
            v += self.PRUNE_KEEP_SEMANTIC_BIAS

        if room["meta"].get("archived"):
            v -= 0.10  # archived rooms are cheaper to prune

        rid = room["id"]
        if rid in self.anchor_ids:
            v += 1.00  # effectively protected

        return v

    def _enforce_capacity(self):
        """
        Keep total room count <= max_rooms by pruning lowest-value rooms.
        Prefers pruning archived first if enabled.
        """
        while len(self.rooms) > self.max_rooms:
            self._prune_one()

    def _prune_one(self):
        if not self.rooms:
            return

        # Do not prune anchors unless absolutely forced
        candidates = [r for r in self.rooms if not (self.PRUNE_PROTECT_ANCHORS and r["id"] in self.anchor_ids)]
        if not candidates:
            # if only anchors exist, prune the oldest anchor (rare / last resort)
            candidates = list(self.rooms)

        if self.PRUNE_ARCHIVE_FIRST:
            archived = [r for r in candidates if r["meta"].get("archived")]
            pool = archived if archived else candidates
        else:
            pool = candidates

        victim = min(pool, key=self._room_value)
        victim_id = victim["id"]

        # remove room
        self.rooms = [r for r in self.rooms if r["id"] != victim_id]

        # remove graph edges
        self.graph.pop(victim_id, None)
        for neigh in self.graph.values():
            neigh.pop(victim_id, None)

        # cleanup anchor set if needed
        self.anchor_ids.discard(victim_id)

    # ──────────────────────────────────────────────────────────────────────────
    # Introspection
    # ──────────────────────────────────────────────────────────────────────────
    def status(self) -> str:
        edges = sum(len(v) for v in self.graph.values()) // 2
        archived = sum(1 for r in self.rooms if r["meta"].get("archived"))
        kinds = Counter(r["meta"]["kind"] for r in self.rooms)
        return (
            f"Martian v0.3 status\n"
            f"  rooms: {len(self.rooms)} / {self.max_rooms}   archived: {archived}\n"
            f"  anchors: {len(self.anchor_ids)}   attractors: {len(self.attractors)}   graph edges: {edges}\n"
            f"  kinds: {dict(kinds)}\n"
            f"  next id: {self.room_id_counter}"
        )


# =============================================================================
#  somnambulist / sleepwalker loop (optional)
#  - periodically dreams (rehearsal sampling)
#  - triggers reflect() when drift / pressure rises
# =============================================================================

class Somnambulist:
    """
    Background process:
      - dream pressure increases over time
      - rehearses by randomly sampling rooms and re-touching them (LRU)
      - triggers Martian.reflect() when pressure crosses threshold
    """

    def __init__(self, martian: Martian,
                 tick_s: float = 1.5,
                 reflect_every_s: float = 10.0,
                 rehearsal_per_tick: int = 2):
        self.m = martian
        self.tick_s = tick_s
        self.reflect_every_s = reflect_every_s
        self.rehearsal_per_tick = rehearsal_per_tick
        self.running = False
        self.dream_level = 0
        self._last_reflect = 0.0

    def step(self):
        self.dream_level += 1

        # rehearsal sampling (touch rooms to bias access_order / keep graph warm)
        active = [r for r in self.m.rooms if not r["meta"].get("archived")]
        if active:
            for _ in range(self.rehearsal_per_tick):
                r = random.choice(active)
                self.m.access_order.append(r["id"])

        # periodic reflect
        now = time.time()
        if now - self._last_reflect >= self.reflect_every_s:
            hub = self.m.reflect()
            self._last_reflect = now
            return hub
        return None

    def run(self, duration_s: float = 20.0):
        self.running = True
        start = time.time()
        while self.running and (time.time() - start) < duration_s:
            hub = self.step()
            if hub is not None:
                print(f"Somnambulist: reflect() produced hub id {hub}")
            time.sleep(self.tick_s)

    def stop(self):
        self.running = False


# ─── Quick demo ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    m = Martian(max_rooms=120)

    # anchors + attractors
    m.add("Julian in Charlotte NC working on infinite memory + epic music.", kind="commitment", is_anchor=True)
    m.add("Maintain phase-stable creative momentum; ship tangible artifacts.", kind="state", attractor=True)

    # episodic fragments
    for i in range(16):
        m.add(f"Episodic fragment {i}: chaotic creative energy, planning, tasks, and shifting focus.", kind="episodic")

    # semantic notes
    m.add("Billie Chihiro: 5+ min desperate epic build with structured crescendo.", kind="semantic")
    m.add("Somnambulist should consolidate episodic noise into semantic hubs with provenance.", kind="semantic")

    print(m.status())
    print("\nTalos:", m.talos_check("Want stable creative momentum tonight; avoid rumination loops."))

    print("\nContext for 'music project':")
    print(m.get_context("ideas for Billie Chihiro track"))

    # background sleepwalker
    s = Somnambulist(m, tick_s=1.5, reflect_every_s=6.0, rehearsal_per_tick=2)
    print("\n— starting somnambulist —")
    s.run(duration_s=14.0)
    print("\n— done —")
    print(m.status())
    print("\nContext after dreaming:")
    print(m.get_context("creative momentum and music build"))
