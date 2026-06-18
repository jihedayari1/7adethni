# Design prompt — paste into Claude (Artifacts) or Google Stitch

> Copy everything in the box below. It asks for a single self-contained, ready-to-use UI for the
> 7adethni browser-extension popup, with an authentic-yet-modern Tunisian identity.

---

**Design a browser-extension popup UI for "7adethni" (حدثني) — a Tunisian Arabizi writing assistant.**
It helps everyday Tunisians write social-media **posts, replies, captions, rewrites and translations**
in natural Tunisian Derja written in Latin letters (Arabizi, e.g. "3aslema, chnowa el a7wel").

Deliver **one self-contained HTML file with inline CSS (vanilla, no external libraries)**, a fixed
popup size **360px wide, ~560px tall**, responsive content, light mode. Clean, friendly, modern —
NOT cluttered, NOT a heavy "ethnic" cliché. Make it feel premium and local.

**Tunisian visual identity (use subtly, not loudly):**
- Palette: warm off-white/cream background (#FAF7F0); primary accent **Tunisian red (#E70013)**;
  a calm **Sidi-Bou-Said Mediterranean blue (#2A6FB0)** as secondary; soft terracotta/sand neutrals;
  a touch of **jasmine** (warm off-white/pale yellow) for highlights.
- A very subtle **zellige/mosaic geometric motif** allowed only as a faint header accent or divider —
  keep it minimal and tasteful.
- Rounded corners (10–14px), soft shadows, generous spacing, large tap targets.
- Typography: a clean modern sans for body; a slightly characterful display weight for the "7adethni"
  wordmark. Must render Latin letters + numbers + emoji well.

**Components (top to bottom):**
1. **Header**: the "7adethni 🇹🇳" wordmark on the left; a small ⚙️ settings icon on the right;
   a tiny **quota chip** ("12/15 free el yom") near the header.
2. **Feature selector**: 6 options as a horizontal scrollable row of pill tabs with small icons —
   *Write a post · Reply · Rewrite · Translate to Derja · Caption · Ask*.
3. **Tone chips**: a wrap of small selectable chips — *3adi · Morfeh 😄 · Rasmi · Promo 🛍️ · 7anin 🤍 · 9sir*.
4. **Input**: a multiline textarea with a friendly Tunisian placeholder ("Ekteb el mawdou3 heni...").
5. **Primary button**: full-width, Tunisian-red, label **"Génère bel tounsi ✨"**; show a loading state.
6. **Output card**: white rounded card showing the generated Arabizi text, with three actions —
   **Copy 📋**, **Regenerate 🔄**, and a small **feedback** pair (👍 good / ✏️ fix) under it.
7. **Footer micro-copy** (tiny, muted): "7adethni — el AI el tounsi".

**Voice / microcopy:** mix Tunisian Arabizi + a little French, the way Tunisians actually talk
(e.g. button "Génère bel tounsi", status "9a3ed ye5dem… ⏳", success "T-copia 📋"). Keep it warm and proud.

**Accessibility:** good contrast, focus states, keyboard usable, aria labels.

Output the complete HTML/CSS as a single artifact I can preview and tweak. Include 2–3 example
generated outputs as placeholder text so the layout is realistic.
```
```
---

## After you get the design
Hand me the produced HTML/CSS and I'll wire it to the backend (feature + tone + input → `POST
/generate`, plus the 👍/✏️ feedback → `POST /feedback`) and replace the current skeleton `extension/popup.*`.
