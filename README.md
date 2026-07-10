# LocalAi

App macOS nativa (SwiftUI) per scaricare e far girare i modelli **Qwen** in locale su Apple Silicon con **MLX Swift** — nessun server esterno: l'inferenza avviene dentro l'app. Include una **modalità agente** stile Claude Code: il modello può esplorare e modificare i file di un progetto ed eseguire comandi, con la tua approvazione.

L'interfaccia riprende lo stile del terminale di Claude Code: tema scuro, font mono, accento arancione, input col prompt `>`, tool call come righe `⏺ tool(...)` con risultato `⎿` espandibile, ragionamento `✻` ripiegabile e approvazioni inline.

## Requisiti

- macOS 14+ su Apple Silicon (M1 o successivo)
- Xcode (per compilare)
- Connessione a internet solo per il primo download di un modello

## Avvio

```sh
./run.sh          # compila, crea LocalAi.app e la apre
./run.sh build    # crea soltanto LocalAi.app
```

La prima compilazione richiede qualche minuto (e una tantum il Metal Toolchain: `xcodebuild -downloadComponent MetalToolchain`). Le successive sono rapide. Il bundle **LocalAi.app** viene creato nella cartella del progetto: puoi trascinarlo in /Applications.

> **Nota**: la build passa da `xcodebuild` (lo fa `run.sh` per te) perché `swift build` da riga di comando non compila gli shader Metal di MLX — il binario partirebbe ma fallirebbe alla prima inferenza con "Failed to load the default metallib". In alternativa puoi aprire il pacchetto in Xcode (`open Package.swift`) e premere Run.

## Come funziona

- **Barra laterale**: catalogo di modelli Qwen (formato MLX, 4 bit, da `mlx-community`). ⬇️ scarica e carica; ▶️ carica un modello già sul disco; 🗑️ lo elimina. I modelli vivono nella cache standard di Hugging Face (`~/.cache/huggingface/hub`), condivisa con Python e la CLI `hf`.
- **Chat**: streaming token per token, velocità in token/s, blocco di ragionamento `<think>` di Qwen3 mostrato a parte e ripiegabile. I **blocchi di codice** nelle risposte sono renderizzati a parte con etichetta del linguaggio e pulsanti **copia** / **salva…** (estensione file scelta in base al linguaggio).
- **Impostazioni** (toolbar): temperatura, prompt di sistema, thinking on/off, modalità agente e auto-approvazione — valgono dalla prossima *Nuova chat*. I **limiti GPU** si applicano invece subito.

## Protezioni di memoria

Un LLM locale senza freni può saturare la RAM e bloccare il Mac. LocalAi ha quattro difese:

| Protezione | Default | Dove |
|---|---|---|
| Tetto rigido memoria GPU (`relaxed: false`: oltre il limite la generazione fallisce con un errore, il sistema resta vivo) | 60% della RAM | Impostazioni |
| Limite cache buffer MLX (senza limite cresce all'infinito) | 64 MB | Impostazioni |
| Massimo token per singola risposta | 4096 | fisso |
| Massimo tool call per turno (contro i loop dei modelli piccoli) | 30 | fisso |

La status bar mostra memoria GPU attiva e picco in tempo reale. All'espulsione di un modello la cache viene svuotata (`GPU.clearCache()`).

## Modalità agente (stile Claude Code)

1. Dalla toolbar scegli un **Workspace** (la cartella del progetto).
2. Il modello riceve questi strumenti, confinati al workspace (8 con l'indice RAG attivo):

| Tool | Cosa fa | Conferma richiesta |
|---|---|---|
| `search_documents` | ricerca semantica sui documenti indicizzati (RAG) | no |
| `read_file` | legge un file con numeri di riga (offset/limit) | no |
| `list_directory` | elenca una directory | no |
| `glob` | trova file per pattern (`*.swift`, `Sources/**/*.swift`) | no |
| `grep` | cerca nei contenuti (regex, `grep -rn`) | no |
| `write_file` | crea/sovrascrive un file | **sì** |
| `edit_file` | sostituzione esatta di una stringa in un file | **sì** |
| `bash` | esegue un comando zsh nel workspace (timeout 120 s) | **sì** |

3. Le chiamate compaiono in chat come card con argomenti e risultato espandibile. Per i tool di scrittura e i comandi shell l'app chiede conferma (Consenti / Nega); in Impostazioni puoi attivare l'auto-approvazione.
4. **Confine di scrittura** (Impostazioni → "scritture solo in…"): puoi limitare `write_file`/`edit_file` a una sottocartella specifica del workspace — la lettura resta libera, ma il modello può creare/modificare file solo lì (es. `output/`). Il system prompt glielo comunica e ogni tentativo fuori confine viene rifiutato con l'indicazione della cartella permessa.
5. Il loop è automatico: il modello chiama un tool → riceve il risultato → continua, fino alla risposta finale.

## Memoria di progetto e contesto (stile CLAUDE.md, /init, /compact)

- **Memoria di progetto**: alla scelta del workspace l'app cerca `LOCALAI.md` (poi `CLAUDE.md`, `AGENTS.md`) nella radice e lo inietta nelle istruzioni di sistema — le tue convenzioni e i comandi del progetto guidano ogni chat. Il banner iniziale mostra quale file è caricato.
- **Genera LOCALAI.md** (come `/init`): se il file non c'è, un clic nel banner fa esplorare il progetto all'agente e gli fa scrivere il file (con la tua approvazione sulla scrittura).
- **Contatore contesto**: la status bar mostra i token usati (`ctx 12.3k/32k`); oltre l'85% diventa arancione e suggerisce di compattare.
- **Compatta** (come `/compact`, in toolbar): il modello riassume la conversazione (obiettivi, decisioni, file toccati, prossimi passi) e la chat riparte con il riassunto nel contesto — la memoria di progetto resta.

Esempi di richieste: *"trova tutti i TODO nel progetto"*, *"leggi Package.swift e spiegami le dipendenze"*, *"aggiungi la gestione errori in ModelStore.swift e compila per verificare"*.

## Generazione immagini (SD/SDXL)

Sezione **"modello immagine"** in sidebar: monti un checkpoint Stable Diffusion / SDXL **single-file** (es. un tuo .safetensors da ComfyUI — validato con l'ispettore integrato) e generi dalla chat con **`/img <descrizione>`**. L'immagine appare nel transcript con **salva…** e **Finder**; i file vanno in `~/Documents/LocalAiImages/`.

Tecnica: worker Python persistente (la pipeline si carica una volta sola) con `diffusers.from_single_file` su GPU Metal (MPS); progresso passo-passo nel transcript e in sidebar. Lo stack (torch+diffusers, ~2,5 GB) si installa on-demand in `Application Support/LocalAi/imagegen` e si rimuove da lì. Primo caricamento del modello: 1–2 minuti; poi ~30–90 s per immagine (SDXL 1024px).

## RAG semantico (ricerca nei tuoi documenti)

Dalla toolbar, pannello **RAG** (✨🔍). Con un workspace scelto, "costruisci indice" fa l'embedding di tutti i documenti (testo, markdown, PDF, codice) con `multilingual-e5-small` — multilingue, eccellente in italiano, ~450 MB una tantum nella cache HF condivisa. L'indice vive in `.localai/rag-index.json` dentro il workspace e si ricarica da solo alle aperture successive.

Con l'indice attivo l'agente riceve il tool **`search_documents`**: trova i passaggi per *significato* (non solo per parola esatta come `grep`) e risponde citando i file di provenienza. Il system prompt gli insegna quando preferirlo a grep. È la risposta giusta a "risponde con precisione su cosa c'è nei miei documenti" — il fine-tuning serve invece per stile e formato.

## Training (fine-tuning LoRA/QLoRA)

Dalla toolbar, pulsante **Training** (🎓). Pipeline in 4 passi, tutto in locale con lo stack ufficiale MLX (`mlx-lm`):

1. **Strumenti**: **integrati nell'app** — il bundle contiene un CPython standalone ricollocabile ([python-build-standalone](https://github.com/astral-sh/python-build-standalone), 3.12) con `mlx-lm` preinstallato in `Contents/Resources/python/`. Nessuna dipendenza esterna: installare = trascinare LocalAi.app, disinstallare = cestinarla. (Se lanci il binario di sviluppo senza bundle c'è un fallback su venv esterno.)
2. **Dati**: importi file grezzi (txt, md, csv, json, pdf). I `.jsonl` già in formato chat (`{"messages": […]}`) vengono validati ed entrano diretti nel dataset.
3. **Preparazione AI**: il modello caricato in chat spezza i file in blocchi e genera esempi di training (domanda/risposta fedeli al testo, JSONL), con split automatico 90/10 train/valid.
4. **Training e fusione**: QLoRA con `mlx_lm.lora` (iterazioni, batch, learning rate regolabili; progresso e loss in tempo reale nel log; interrompibile), poi `mlx_lm.fuse` produce un modello autonomo e **"carica in chat"** lo apre subito nell'app.

**Export per sistemi live** (passo 5 della finestra Training): **fp16** produce safetensors HF standard dequantizzati, caricabili direttamente da vLLM/TGI/transformers su server Linux/GPU; **GGUF** per Ollama/llama.cpp (diretto solo per le architetture supportate da mlx-lm; per le altre: fp16 + `convert_hf_to_gguf.py` di llama.cpp). In alternativa, la cartella fusa MLX si serve così com'è con `mlx_lm.server` su un Mac (API compatibile OpenAI).

I progetti di training vivono in `~/Documents/LocalAiTraining/<progetto>/` (data/, adapters/, fused/, export-fp16/, export-gguf/). I modelli fusi sono normali cartelle MLX riutilizzabili ovunque. Consiglio: parti dal 4B, 500–1000 iterazioni, e tieni d'occhio la val loss (se sale stai overfittando).

Cosa resta sul disco oltre all'app (dati dell'utente, non dipendenze): i modelli scaricati (`~/.cache/huggingface/hub`, eliminabili dalla sidebar) e i progetti di training (`~/Documents/LocalAiTraining`).

## Catalogo modelli

| Modello | Dimensione | Note |
|---|---|---|
| Qwen3 0.6B | ~350 MB | test rapidissimo |
| Qwen3 1.7B | ~1 GB | velocissimo |
| Qwen3 4B | ~2.3 GB | **il più piccolo affidabile come agente** (smoke test: PASS) |
| Qwen3 8B | ~4.7 GB | uso quotidiano |
| Qwen3 14B | ~8.4 GB | alta qualità |
| Qwen3 30B A3B (MoE) | ~17 GB | qualità 30B, velocità 3B — il migliore come agente |
| Qwen3 32B | ~18 GB | il più capace |
| Qwen2.5 Coder 7B | ~4.3 GB | specializzato in codice (senza thinking) |

Con 48 GB di RAM girano tutti. Per la modalità agente conviene un modello ≥ 4B: i più piccoli sbagliano spesso le tool call.

**Altri modelli (sezione "altri modelli" in sidebar)**: agganci qualsiasi repo Hugging Face in formato MLX scrivendo l'id (es. `mlx-community/Llama-3.2-3B-Instruct-4bit`) o una **cartella-modello locale** (es. un tuo fine-tuning fuso). La lista è persistente; "Rimuovi dalla lista" dal menu contestuale.

**Validazione al montaggio**: se la cartella scelta non è un LLM caricabile (manca config.json, è una pipeline diffusers, niente safetensors) l'app lo spiega subito con il motivo. **Ispettore safetensors** ("ispeziona safetensors…" in sidebar): identifica qualsiasi file .safetensors senza caricarlo — checkpoint o LoRA, architettura (SDXL, Flux, Wan/DiT, LLM…), parametri, e dove va messo.

**Allegati in chat**: 📎 nell'input o **drag & drop** nel transcript — txt, md, csv, json e PDF vengono estratti e passati al modello col messaggio (max ~12k caratteri per file). Utile per "analizza questo documento" senza workspace né indice.

## Smoke test

Verifica end-to-end del loop agentico (usa Qwen3 4B in un workspace temporaneo — 0.6B/1.7B vanno in loop sulle tool call: lettura e scrittura file via tool; fusibili: 12 tool call max, timeout 6 min, limiti GPU):

```sh
./run.sh smoke
```

## Architettura

| Target / File | Ruolo |
|---|---|
| `QwenLocalCore/AgentToolbox.swift` | i 7 tool dell'agente, confinati al workspace |
| `QwenLocalCore/ModelCatalog.swift` | catalogo curato dei modelli Qwen |
| `LocalAi/QwenEngine.swift` | download/caricamento modello, sessione, streaming, dispatch tool + approvazioni |
| `LocalAi/ModelStore.swift` | scansione e pulizia della cache Hugging Face |
| `LocalAi/ContentView.swift` | sidebar con gestione modelli |
| `LocalAi/ChatView.swift` | transcript stile terminale, righe tool, approvazioni inline, impostazioni |
| `LocalAi/Theme.swift` | palette scura e tipografia mono stile Claude Code |
| `LocalAi/TrainingManager.swift` | pipeline fine-tuning: runtime integrato, preparazione dati AI, LoRA, fusione |
| `LocalAi/RAGManager.swift` | indice semantico del workspace (MLXEmbedders, e5-small) e ricerca per coseno |
| `LocalAi/ImageGenManager.swift` | generazione immagini SD/SDXL: worker diffusers persistente su MPS |
| `LocalAi/SafetensorsInspector.swift` | identificazione file .safetensors dall'intestazione |
| `LocalAi/TrainingView.swift` | finestra Training a 4 passi con log live |
| `QwenSmoke/Smoke.swift` | smoke test CLI del loop agentico |

Librerie: [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) (`MLXLLM`, `MLXLMCommon`, `MLXHuggingFace`), [swift-huggingface](https://github.com/huggingface/swift-huggingface), [swift-transformers](https://github.com/huggingface/swift-transformers).

## Cosa manca rispetto a Claude Code

Per onestà: niente sotto-agenti, MCP, hooks o permessi granulari per comando. Il limite principale è il modello: un LLM locale da 4–30B è molto meno affidabile come agente. Buoni prossimi passi: persistenza delle conversazioni, diff-preview prima di approvare una modifica, allowlist di comandi bash.
