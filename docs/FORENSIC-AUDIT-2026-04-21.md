# Neural Memory Forensic Audit — 2026-04-21

## "Warum funktioniert Neural Memory nicht, wenn es installiert ist?"

Eine vollständige forensische Analyse aller Failure Modes, die am 21.04.2026
identifiziert, debuggt und (teilweise) gefixt wurden. Dokumentiert als
Post-Mortem, nicht als Feature-Dokumentation.

---

## Executive Summary

Neural Memory hat 7 unabhängige Failure Modes, die alle gleichzeitig
auftreten können. Keiner davon ist "Neural Memory ist kaputt" — sie sind
alle subtil, alle versteckt, und alle erzeugen das gleiche Symptom:
**"Es tut nicht, was es soll."**

Die meisten haben ihre Wurzel in einem Grundproblem:
**Neural Memory wurde als Plugin nachträglich in eine Architektur eingebaut,
die nie für Plugins dieser Komplexität designed wurde.**

---

## Failure Mode #1: Tool-Routing — "Unknown tool: neural_remember"

### Symptom
Der Agent sieht `neural_remember` in seiner Tool-Liste, ruft es auf,
bekommt `{"error": "Unknown tool"}` zurück. Direkte Python-API-Aufrufe
funktionieren. Nur der Agent-Path ist kaputt.

### Root Cause Chain
```
1. run_agent.py:1184  →  injiziert Schemas in self.tools (LLM sieht die Tools)  ✅
2. run_agent.py:1190  →  fügt zu valid_tool_names hinzu  ✅
3. LLM ruft neural_remember auf  →  model_tools.py handle_function_call()
4. handle_function_call()  →  registry.dispatch("neural_remember")
5. registry kennt neural_remember NICHT  →  {"error": "Unknown tool"}  ❌
6. MemoryManager.handle_tool_call() wird NIE aufgerufen  ❌
```

### Warum passiert das?
Die Hermes-Architektur hat zwei parallele Tool-Systeme:
- **Tool Registry**: Altes System, `registry.register()` → `registry.dispatch()`
- **Memory Manager**: Neues System, Memory-Provider haben `get_tool_schemas()` + `handle_tool_call()`

Neural Memory registriert sich im Memory Manager. Aber `handle_function_call()`
in `model_tools.py` routet ALLES durch die Registry. Die Registry weiß
nichts von Neural Memory. Resultat: Dead End.

### Fix
`model_tools.py` bekommt eine Modul-Variable `_memory_manager_ref` und eine
Setter-Funktion `set_memory_manager()`. `handle_function_call()` prüft den
Memory Manager VOR der Registry:

```python
if _memory_manager_ref and _memory_manager_ref.has_tool(function_name):
    result = _memory_manager_ref.handle_tool_call(function_name, function_args)
else:
    result = registry.dispatch(function_name, function_args, ...)
```

`run_agent.py` ruft `_smm(self._memory_manager)` nach der Tool-Injection auf.

### Status: ✅ GEFIXT

---

## Failure Mode #2: Conflict Detection — "neural_remember gibt immer ID 1256 zurück"

### Symptom
Egal was man speichert — `neural_remember` gibt immer die gleiche ID zurück.
Direkte SQL-INSERTs funktionieren. Direkte `NeuralMemory.remember()`-Aufrufe
funktionieren. Nur über den Tool-Path nicht.

### Root Cause
`memory_client.py` hat `detect_conflicts=True` als Default. Conflict Detection
vergleicht Embeddings via Cosine Similarity. Mit dem Hash-Backend (Fallback
wenn FastEmbed nicht verfügbar) produziert jeder Vektor ähnliche Werte —
Fake-Similarity > 0.7 triggert bei JEDEM neuen Memory einen "Conflict".

Der Conflict Handler supersedet ein existierendes Memory und gibt dessen
ID zurück. Also immer die gleiche ID, nie ein neues Memory.

### Warum so schwer zu debuggen?
1. Direkte Python-Aufrufe nutzen denselben Code — aber man testet mit
   `detect_conflicts=False` weil man es "weiß"
2. Der Agent nutzt den Default (`True`) — und das Hash-Backend ist subtil
3. Man denkt "Tool-Routing ist kaputt" (#1), nicht "Embedding produziert Müll"
4. Selbst wenn man #1 fixt, bleibt #2 bestehen — andere Symptom, gleiche Wahrnehmung

### Fix
```python
_reliable_backends = {'FastEmbedBackend', 'SentenceTransformerBackend'}
_backend_name = type(self.embedder.backend).__name__
_can_detect = _backend_name in _reliable_backends

if detect_conflicts and self._graph_nodes and _can_detect:
    conflicts = self._find_conflicts(text, embedding)
```

Hash-Backend triggert nie Conflict Detection.

### Status: ✅ GEFIXT

---

## Failure Mode #3: Embedding Backend — "FastEmbed wird nicht geladen"

### Symptom
Config sagt `embedding_backend: fastembed`, aber Recall liefert Müll
(Similarity ~0.07 statt ~0.8+). Embeddings haben Dimension 384 statt 1024.

### Root Cause
`embed_provider.py` existiert in ZWEI Versionen:
- **hermes-agent** (915 Zeilen): Hat `FastEmbedBackend` ✅
- **neural-memory-adapter source** (875 Zeilen): Hat KEIN `FastEmbedBackend` ❌

Unbekannte Backend-Namen fallen auf `HashBackend` durch — Random-Embeddings,
keine echte Semantik. Die Config sagt "fastembed", der Code sagt "kenn ich nicht,
hier sind Zufallszahlen".

### Warum so schwer zu debuggen?
1. Man prüft die Config — die sagt "fastembed" ✅
2. Man prüft die Datei im neural-memory-adapter — da steht "875 Zeilen"
3. Man denkt "875 Zeilen, da müsste FastEmbed drin sein" — ist es aber nicht
4. Die DEPLOYED Version (915 Zeilen) hat es — aber die SOURCE nicht
5. Beim nächsten Sync überschreibt die SOURCE die DEPLOYED → FastEmbed ist weg

### Fix
`FastEmbedBackend` Klasse in `embed_provider.py` hinzugefügt:
- Nutzt `fastembed.TextEmbedding` mit `intfloat/multilingual-e5-large`
- 1024d Embeddings, ONNX Runtime, ~50ms pro Embedding
- In `_auto_detect()` Priority-Chain eingefügt

### Aktueller Stand
- **Deployed** (hermes-agent): 915 Zeilen, FastEmbed=True ✅
- **Source** (neural-memory-adapter): 875 Zeilen, FastEmbed=False ❌
- **MISMATCH** — Source ist outdated!

### Status: ⚠️ TEILWEISE — Deployed OK, Source nicht sync'd

---

## Failure Mode #4: GPU Engine DB Isolation — "Phantom IDs aus Production-DB"

### Symptom
`neural_recall` auf einer Test-DB (`/tmp/test.db`) gibt IDs zurück, die
in der Test-DB nicht existieren. Kommen aus der Production-DB
(`~/.neural_memory/memory.db`).

### Root Cause
`gpu_recall.py` hardcoded `_CACHE_DIR = Path.home() / ".neural_memory" / "gpu_cache"`.
`NeuralMemory.__init__` erstellt immer einen `GpuRecallEngine`, egal welche
`db_path` angegeben wurde. Der GPU-Engine lädt aus dem Production-Cache
und umgeht die angegebene DB komplett.

### Warum so schwer zu debuggen?
1. Test erstellt saubere DB → speichert Memory → recall → kriegt falsche IDs
2. Man denkt "Embedding ist kaputt" oder "Cosine Similarity ist falsch"
3. Tatsächlich ist der Code-Weg: `recall()` → GPU-Engine → Production-Cache
4. SQLite-DB wird gar nicht angefasst — aber man prüft die SQLite-DB
5. Die SQLite-DB ist sauber, die Ergebnisse sind trotzdem falsch

### Fix
```python
self._gpu = None
if db_path == DB_PATH:  # nur mit Standard-DB laden
    try:
        from gpu_recall import GpuRecallEngine
        self._gpu = GpuRecallEngine()
    except Exception:
        self._gpu = None
```

### Status: ✅ GEFIXT

---

## Failure Mode #5: Embedder Double-Loading — "FastEmbed wird zweimal geladen"

### Symptom
Beim Start zwei Mal `[embed] FastEmbed loaded:` im Log. ~500MB Modell
doppelt im RAM. Startup braucht 2x so lang.

### Root Cause
```
Memory.__init__()       →  erstellt EmbeddingProvider()        →  FastEmbed Ladevorgang #1
  ↓
NeuralMemory.__init__() →  erstellt NOCHMAL EmbeddingProvider() →  FastEmbed Ladevorgang #2
```

`NeuralMemory` nimmt keinen `embedder=` Parameter — es erstellt immer
seinen eigenen. `Memory` (Wrapper) erstellt auch einen. Zwei Instanzen,
doppelter Speicher.

### Fix
`NeuralMemory.__init__` akzeptiert optional `embedder=None`:
```python
def __init__(self, ..., embedder=None):
    if embedder is not None:
        self.embedder = embedder
    else:
        from embed_provider import EmbeddingProvider
        self.embedder = EmbeddingProvider(backend=embedding_backend)
```

`Memory.__init__` übergibt `self._embedder` an `NeuralMemory`.

### Status: ✅ GEFIXT

---

## Failure Mode #6: __init__.py _load_config() — "name '_load_config' is not defined"

### Symptom
Plugin lädt, Provider wird initialisiert, dann:
`NameError: name '_load_config' is not defined`

### Root Cause
`hermes-plugin/__init__.py` (1116 Zeilen, alter Merge-Artifact) rief
`_load_config()` an Zeile 586 auf. Die Funktion existiert nicht — sie
heißt `get_config()` und kommt aus `config.py`.

Die hermes-agent Version (821 Zeilen) nutzt korrekt `get_config()`.

### Warum so schwer zu debuggen?
1. Man denkt "Plugin ist korrekt installiert" — Datei ist da
2. Man denkt "Code ist aktuell" — 1116 Zeilen, sieht viel aus
3. Aber 1116 Zeilen = alter Stand mit Duplikaten und totem Code
4. Die 821-Zeilen-Version ist die aktuelle — aber nicht deployed

### Fix
`__init__.py` von hermes-agent synced nach neural-memory-adapter:
```bash
cp ~/.hermes/hermes-agent/plugins/memory/neural/__init__.py \
   ~/projects/neural-memory-adapter/hermes-plugin/__init__.py
```

### Aktueller Stand
- Deployed: ✅ `get_config()` (korrekt)
- Source: ✅ Synced
- Duplicate class: ✅ Keine
- `_dream` vs `_dream_engine`: ✅ Nur `_dream`

### Status: ✅ GEFIXT

---

## Failure Mode #7: Cross-Repo Sync — "Drei Kopien, eine Wahrheit"

### Symptom
Änderungen an neural-memory-Dateien wirken nicht. Tests bestehen,
Runtime verhält sich anders. Oder umgekehrt.

### Root Cause
Neural Memory Dateien existieren in 3+ Orten:
```
1. ~/projects/neural-memory-adapter/python/         (Source, standalone testbar)
2. ~/projects/neural-memory-adapter/hermes-plugin/  (Deployment-Artifact)
3. ~/.hermes/hermes-agent/plugins/memory/neural/     (Deployed, wird vom Agent genutzt)
```

Dazu optional:
```
4. ~/.hermes/plugins/memory/neural/                  (Legacy-Location?)
5. ~/.hermes/tools/neural_tools.py                   (Tool-Registration, separat)
```

Wenn man nur eine Kopie ändert, funktioniert es in einem Kontext
aber nicht in anderen. Man testet im falschen Kontext, denkt "funktioniert",
und wundert sich warum der Agent es nicht sieht.

### Konkreter Vorfall (21.04.2026)
- `embed_provider.py` in hermes-agent: 915 Zeilen (mit FastEmbed) ✅
- `embed_provider.py` in neural-memory-adapter source: 875 Zeilen (ohne FastEmbed) ❌
- Beim nächsten `cp python/ hermes-plugin/` wäre FastEmbed verschwunden

### Fix
**Nicht implementiert** — es gibt keine automatische Sync-Maschinerie.
Manuell: `cp` zwischen den Orten. Skill `neural-memory-file-sync` dokumentiert
das Muster, aber es wird nicht enforced.

### Status: ❌ UNRESOLVIERT — Fundamentales Architekturproblem

---

## Database Zustand (21.04.2026, 21:15)

```
Speicherort:    ~/.neural_memory/memory.db
Memories:       1377
Connections:    135143
NULL Embeddings: 0
Self-Loops:     0
Embedding Dim:  1024d
Magnitude:      1.0000 (normalisiert, korrekt)

Benchmark-Garbage:
  DD* Labels:    0
  turn-* Labels: 56 (4.1% — Auto-Saved Conversation Noise)
  Session-summaries: 61

Embedding Backend: FastEmbed (intfloat/multilingual-e5-large)
C++ Bridge:        Deaktiviert (use_cpp=False)
GPU Engine:        Nur mit Standard-DB geladen (isoliert)
```

---

## Was NOCH nicht gefixt ist

### 1. Source/Deployed Mismatch (embed_provider.py)
Die Source in `neural-memory-adapter` hat nicht das FastEmbed-Backend.
Beim nächsten Sync wird die Deployed-Version überschrieben.

### 2. 56 Auto-Saved "turn-" Einträge
Nicht kritisch, aber Müll im Graph — Auto-gespeicherte Konversations-Rohdaten
inklusive ungefilterter User-Nachrichten. Sollten gelöscht werden.

### 3. Kein automatischer File-Sync
Manuelles `cp` zwischen 3 Orten ist fehleranfällig. Ein Git-Hook oder
Sync-Script würde die meisten dieser Probleme verhindern.

### 4. `neural_tools.py` existiert lokal (~/.hermes/tools/) aber nicht in hermes-agent
Aktuell nicht benötigt (MemoryManager-Route funktioniert), aber wenn jemand
den alten "Tool-File" Ansatz folgt, wird er es suchen und nicht finden.

### 5. Dream Engine Race Condition (MSSQL)
`connection_history` unique Index erzeugt Duplicate-Key-Errors bei
parallelen Dream-Cycles. Bereits mit `logger.debug()` gehandhabt —
harmless, aber unschön.

---

## Root Cause Analysis — Warum Neural Memory "seine Arbeit verweigert"

### Schicht 1: Architektur (Fundament)
Neural Memory wurde als Plugin für eine Plattform gebaut, die Plugins
dieser Komplexität nicht vorsieht. Zwei parallele Tool-Systeme
(Registry vs MemoryManager), kein klarer Extension-Point für Memory-Provider.

### Schicht 2: Embedding (Semantische Ebene)
Ohne echte Embeddings (FastEmbed) produziert Neural Memory Zufall.
Der Hash-Backend-Fallback ist eine Zeitbombe — er funktioniert
"irgendwie", aber die Similarity-Werte sind Müll. Conflict Detection,
Auto-Connect, Dream Engine — alles basiert auf Embedding-Quality.

### Schicht 3: Tool-Routing (Dispatch-Ebene)
Selbst wenn Embeddings funktionieren, kommen die Tool-Aufrufe nicht
durch. Die LLM-Seite sieht die Tools, aber der Code-Weg endet in
einer Registry, die nichts von ihnen weiß.

### Schicht 4: DB-Isolation (Daten-Ebene)
Selbst wenn Tools und Embeddings funktionieren, kann der GPU-Engine
Daten aus der falschen DB liefern. Tests sind nicht isoliert.

### Schicht 5: File-Sync (Deployment-Ebene)
Selbst wenn alles lokal funktioniert, kann der nächste Deploy-Sync
eine kaputte Datei über eine funktionierende stülpen.

### Zusammenfassung
**Neural Memory hat nicht EINEN Bug. Es hat eine Kette von 7
unabhängigen Bugs, die sich gegenseitig maskieren. Fixt man #1,
stolpert man über #2. Fixt man #2, tritt #3 auf. Das System ist
nicht "kaputt" — es ist untertestet, undersynced, und unterdocumented
in den Bereichen, die am meisten weh tun.**

---

## Empfehlungen

1. **Automatischer File-Sync** — Git-Hook oder CI, der bei Änderungen
   in `python/` automatisch nach `hermes-plugin/` synced und deployed.

2. **Embedding Backend Health Check** — Startup-Check, der verifiziert,
   dass der konfigurierte Backend auch tatsächlich geladen wurde.
   Warnung wenn Hash-Backend aktiv (außer explizit angefordert).

3. **Test-DB Isolation** — Jeder Test bekommt eine eigene Temp-DB.
   Kein Test darf die Production-DB anfassen. Das ist jetzt schon
   im Test-Suite implementiert, aber nicht enforced.

4. **Single Source of Truth** — Entweder `neural-memory-adapter` ODER
   `hermes-agent` ist die Wahrheit. Nicht beide. Aktuell ist
   `hermes-agent` die Wahrheit, aber `neural-memory-adapter` hat
   eine "alte" Kopie.

5. **Integration Tests** — Nicht Unit-Tests für `NeuralMemory`, sondern
   End-to-End-Tests: "Agent ruft neural_remember auf → Memory wird
   gespeichert → neural_recall findet es → korrekte ID zurück."

---

*Erstellt: 2026-04-21 21:15 MESZ*
*Quellen: 4 Skills (neural-memory-debugging, neural-memory-first,
neural-memory-plugin-architecture, neural-memory-adapter-fix),
Live-Code-Inspection, DB-Audit, Session-Search*
